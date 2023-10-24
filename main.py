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
from model.utils import model_embed,search_top,ParseOrbitrap
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
    
    train_data,word2idx = make_train_data(msms1,precursor1,maxlen)
    
    vocab_size = len(word2idx)
    input_ids, intensity = zip(*train_data) 
    intensity = [torch.FloatTensor(i) for i in intensity] 
    
    model = MSBERT(vocab_size, hidden, n_layers, attn_heads, dropout,maxlen,max_pred)
    model,train_loss,val_loss = TrainMSBERT(model,input_ids,intensity,batch_size,epochs,lr,temperature)
    torch.save(model.state_dict(),'E:/MSBERT_model/1012two_stage/mask.pkl')
    
    train_ref_arr =  model_embed(model,train_ref,batch_size)
    train_query_arr = model_embed(model,train_query,batch_size)
    top = search_top(train_ref_arr,train_query_arr,smiles1,smiles2,batch=50)
    
    test_ref_arr = model_embed(model,test_ref,batch_size)
    test_query_arr = model_embed(model,test_query,batch_size)
    dataset_arr = np.vstack((train_ref_arr,test_ref_arr))
    smiles_list = smiles1+smiles3
    top2 = search_top(dataset_arr,test_query_arr,smiles_list,smiles4,batch=50)
    
    
    


























