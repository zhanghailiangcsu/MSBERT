# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 21:41:00 2023

@author: Administrator
"""
from model.utils import ModelEmbed,ParseOrbitrap
from model.utils import ModelEmbed
import torch
from model.MSBERTModel import MSBERT
from sklearn.cluster import Birch
from sklearn import metrics
from sklearn.metrics import rand_score,adjusted_rand_score,homogeneity_score
from sklearn.metrics import homogeneity_completeness_v_measure,completeness_score
import numpy as np
from Spec2VecModel.TrainSpec2Vec import peak_embed
from data.ProcessData import MakeTrainData

def BirchCluster(n_clusters,test_arr):
    brc = Birch(threshold=0.5,n_clusters=n_clusters)
    brc.fit(test_arr)
    return brc

def GenRawLabel(smiles_unique,smiles):
    smi = smiles_unique[0]
    raw_label = np.zeros(len(smiles))
    l = 0
    for smi in smiles_unique:
        idx = [i for i,v in enumerate(smiles) if v == smi]
        raw_label[idx] = l
        l += 1
    return raw_label

def CalEvaluate(labels_true, labels_pred):
    ari = metrics.adjusted_rand_score(labels_true, labels_pred)
    homogeneity = metrics.homogeneity_score(labels_true, labels_pred)
    completeness = metrics.completeness_score(labels_true, labels_pred)
    v_measure = metrics.v_measure_score(labels_true, labels_pred, beta=0.5)
    result = {'ARI':ari,'homogeneity':homogeneity,
              'completeness':completeness,'v_measure':v_measure}
    return result

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
    
    test_ref2,word2idx = MakeTrainData(msms3,precursor3,100)
    test_arr = ModelEmbed(MSBERTmodel,test_ref2,batch_size)
   
    smiles_unique = list(set(smiles3))
    labels_true = GenRawLabel(smiles_unique,smiles3)
    
    n_clusters = len(smiles_unique)
    brc = BirchCluster(n_clusters,test_arr)
    labels_pred = brc.predict(test_arr)
    # smiles = smiles3
    result = CalEvaluate(labels_true, labels_pred)
    model_file = 'Spec2vecModel/ob_all.model'
    spectrums = [s for s1 in test_ref for s in s1]
    spec2vec_vec = peak_embed(model_file,spectrums,n_decimals=2)
    spec2vec_vec = np.array(spec2vec_vec)
    
    

