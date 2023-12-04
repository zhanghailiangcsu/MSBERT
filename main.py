# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 09:04:35 2023

@author: Administrator
"""
import os
os.chdir('E:/github/MSBERT')
import pickle
from data.LoadGNPS import ProDataset,MakeDataset
from data.ProcessData import MakeTrainData,MakeTestData
import torch
from model.MSBERTModel import MSBERT
from model.Train import  TrainMSBERT
from model.utils import ModelEmbed,SearchTop,ParseOrbitrap,CalMSBERTTop
from model.utils import ParseOtherData,CalCosineTop
import numpy as np
from Spec2VecModel.TrainSpec2Vec import CalSpec2VecTop
import gensim
from tqdm import tqdm

if __name__ == '__main__':
    
    maxlen = 100
    batch_size = 32
    dropout = 0
    hidden=512
    n_layers = 6
    attn_heads = 16
    max_pred = 2
    epochs = 4
    lr = 0.0003
    temperature = 0.005
    
    train_ref,msms1,precursor1,smiles1 = ParseOrbitrap('GNPSdata/ob_train_ref.pickle')
    train_query,msms2,precursor2,smiles2 =ParseOrbitrap('GNPSdata/ob_train_query.pickle')
    test_ref,msms3,precursor3,smiles3 = ParseOrbitrap('GNPSdata/ob_test_ref.pickle')
    test_query,msms4,precursor4,smiles4 = ParseOrbitrap('GNPSdata/ob_test_query.pickle')
    
    train_ref2,word2idx = MakeTrainData(msms1,precursor1,100)
    train_query2,word2idx = MakeTrainData(msms2,precursor2,100)
    test_ref2,word2idx = MakeTrainData(msms3,precursor3,100)
    test_query2,word2idx = MakeTrainData(msms4,precursor4,100)
    
    vocab_size = len(word2idx)
    input_ids, intensity = zip(*train_ref2) 
    intensity = [torch.FloatTensor(i) for i in intensity] 
    input_ids = [torch.LongTensor(i) for i in input_ids] 
    
    MSBERTmodel = MSBERT(vocab_size, hidden, n_layers, attn_heads, dropout,maxlen,max_pred)
    MSBERTmodel,train_loss,val_loss = TrainMSBERT(MSBERTmodel,input_ids,intensity,batch_size,epochs,lr,temperature)
    # torch.save(MSBERTmodel.state_dict(),'E:/MSBERT_model/temperature/0005.pkl')
    MSBERTmodel.load_state_dict(torch.load('E:/MSBERT_model/1025/MSBERT.pkl'))
    
    top = CalMSBERTTop(MSBERTmodel,train_ref2,train_query2,smiles1,smiles2)
    ref_list = train_ref2+test_ref2
    smiles_list = smiles1+smiles3
    top2 = CalMSBERTTop(MSBERTmodel,ref_list,test_query2,smiles_list,smiles4)
    
    model_file = 'Spec2vecModel/ob_all.model'
    Spec2VecModel = gensim.models.Word2Vec.load(model_file)
    Spec2VecObTop1 = CalSpec2VecTop(Spec2VecModel,train_ref,train_query)
    CosineObTop1 = CalCosineTop(train_ref,train_query)
    Spec2VecObTop2 = CalSpec2VecTop(Spec2VecModel,ref_list,test_query)
    CosineObTop2 = CalCosineTop(ref_list,test_query)
    
    with open('GNPSdata/qtof.pickle', 'rb') as f:
        qtof = pickle.load(f)
    ref_data,query_data,smile_ref,smile_query = ParseOtherData(qtof)
    MSBERTQtofTop = CalMSBERTTop(MSBERTmodel,ref_data,query_data,smile_ref,smile_query)
    qtof_ref,qtof_query,_,_ = MakeDataset(qtof,n_max=99,test_size=0,n_decimals=2)
    Spec2VecQtofTop = CalSpec2VecTop(Spec2VecModel,qtof_ref,qtof_query)
    CosineTop = CalCosineTop(qtof_ref,qtof_query)
    
    with open('GNPSdata/other.pickle', 'rb') as f:
        other = pickle.load(f)
    ref_data,query_data,smile_ref,smile_query = ParseOtherData(other)
    MSBERTOtherTop = CalMSBERTTop(MSBERTmodel,ref_data,query_data,smile_ref,smile_query)
    other_ref,other_query,_,_ = MakeDataset(other,n_max=99,test_size=0,n_decimals=2)
    Spec2VecOtherTop = CalSpec2VecTop(Spec2VecModel,other_ref,other_query)
    CosineTop = CalCosineTop(other_ref,other_query)
    
    
    


























