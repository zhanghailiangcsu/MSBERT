# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 09:04:35 2023

@author: Administrator
"""
import os
os.chdir('E:/github/MSBERT')
import pickle
from data.LoadGNPS import pro_dataset
from data.ProcessData import make_train_data,make_test_data
import torch
from model.MSBERTModel import MSBERT
from model.Train import  TrainMSBERT
from model.utils import ModelEmbed,SearchTop,ParseOrbitrap,CalMSBERTTop
from model.utils import ParseOtherData
import numpy as np


if __name__ == '__main__':
    
    maxlen = 100
    batch_size = 32
    dropout = 0
    hidden=512
    n_layers = 6
    attn_heads = 16
    max_pred = 3
    epochs = 1
    lr = 0.0003
    temperature = 0.01
    
    train_ref,msms1,precursor1,smiles1 = ParseOrbitrap('GNPSdata/ob_train_ref.pickle')
    train_query,msms2,precursor2,smiles2 =ParseOrbitrap('GNPSdata/ob_train_query.pickle')
    test_ref,msms3,precursor3,smiles3 = ParseOrbitrap('GNPSdata/ob_test_ref.pickle')
    test_query,msms4,precursor4,smiles4 = ParseOrbitrap('GNPSdata/ob_test_query.pickle')
    
    train_ref,word2idx = make_train_data(msms1,precursor1,100)
    train_query,word2idx = make_train_data(msms2,precursor2,100)
    test_ref,word2idx = make_train_data(msms3,precursor3,100)
    test_query,word2idx = make_train_data(msms4,precursor4,100)
    train_data,word2idx = make_train_data(msms1,precursor1,maxlen)
    
    vocab_size = len(word2idx)
    input_ids, intensity = zip(*train_data) 
    intensity = [torch.FloatTensor(i) for i in intensity] 
    
    model = MSBERT(vocab_size, hidden, n_layers, attn_heads, dropout,maxlen,max_pred)
    model,train_loss,val_loss = TrainMSBERT(model,input_ids,intensity,batch_size,epochs,lr,temperature)
    torch.save(model.state_dict(),'E:/MSBERT_model/1012two_stage/mask.pkl')
    
    top = CalMSBERTTop(model,train_ref,train_query,smiles1,smiles2)
    
    ref_list = train_ref+test_ref
    smiles_list = smiles1+smiles3
    top2 = CalMSBERTTop(model,ref_list,test_query,smiles_list,smiles4)
    
    with open('GNPSdata/qtof.pickle', 'rb') as f:
        qtof = pickle.load(f)
    ref_data,query_data,smile_ref,smile_query = ParseOtherData(qtof)
    MSBERTQtofTop = CalMSBERTTop(model,ref_data,query_data,smile_ref,smile_query)
    
    with open('GNPSdata/other.pickle', 'rb') as f:
        other = pickle.load(f)
    ref_data,query_data,smile_ref,smile_query = ParseOtherData(other)
    MSBERTOtherTop = CalMSBERTTop(model,ref_data,query_data,smile_ref,smile_query)
    
    


























