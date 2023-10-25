# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 21:41:00 2023

@author: Administrator
"""
from model.utils import ModelEmbed,ParseOrbitrap
from model.utils import ModelEmbed
import torch
from MSBERTModel import MSBERT
from sklearn.cluster import Birch



if __name__ == '__main__':
    maxlen = 100
    batch_size = 32
    dropout = 0
    hidden=512
    n_layers = 6
    attn_heads = 16
    max_pred = 3
    vocab_size = 100002
    
    test_ref,msms3,precursor3,smiles3 = ParseOrbitrap('GNPSdata/ob_test_ref.pickle')
    MSBERTmodel = MSBERT(vocab_size, hidden, n_layers, attn_heads, dropout,maxlen,max_pred)
    MSBERTmodel.load_state_dict(torch.load('E:/MSBERT_model/1025/MSBERT.pkl'))
    test_arr = ModelEmbed(MSBERTmodel,test_ref,batch_size)
    smiles_unique = list(set(smiles3))
    
    brc = Birch(n_clusters=len(smiles_unique))
    brc.fit(test_arr)
    

