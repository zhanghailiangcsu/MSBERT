# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 21:41:00 2023

@author: Administrator
"""
from model.utils import ModelEmbed,ParseOrbitrap
import torch
from model.MSBERTModel import MSBERT
from sklearn.cluster import Birch
from sklearn import metrics
import numpy as np
from Spec2VecModel.TrainSpec2Vec import peak_embed
from data.ProcessData import MakeTrainData
from data.MS2Vec import ms_to_vec

def MSBERTCluster(model_file,test_data,msms,precursor,smiles):
    MSBERTmodel = MSBERT(100002, 512, 6, 16, 0,100,3)
    MSBERTmodel.load_state_dict(torch.load(model_file))
    test_data2,word2idx = MakeTrainData(msms,precursor,100)
    test_arr = ModelEmbed(MSBERTmodel,test_data2,batch_size)
    smiles_unique = list(set(smiles))
    labels_true = GenRawLabel(smiles_unique,smiles)
    brc = BirchCluster(len(smiles_unique),test_arr)
    labels_pred = brc.predict(test_arr)
    msbert_result = CalEvaluate(labels_true, labels_pred)
    return msbert_result

def Spec2VecCluster(model_file,test_data,smiles):
    smiles_unique = list(set(smiles))
    labels_true = GenRawLabel(smiles_unique,smiles)
    spectrums = [s for s1 in test_data for s in s1]
    spec2vec_vec = peak_embed(model_file,spectrums,n_decimals=2)
    spec2vec_vec = np.array(spec2vec_vec)
    brc_spec = BirchCluster(len(smiles_unique),spec2vec_vec)
    labels_pred = brc_spec.predict(spec2vec_vec)
    spec2vec_result = CalEvaluate(labels_true, labels_pred)
    return spec2vec_result

def MSMSCluster(test_data,smiles):
    smiles_unique = list(set(smiles))
    labels_true = GenRawLabel(smiles_unique,smiles)
    spectrums = [s for s1 in test_data for s in s1]
    peaks_vec = MSMS2Vec(spectrums)
    brc_msms = BirchCluster(len(smiles_unique),peaks_vec)
    labels_pred = brc_msms.predict(peaks_vec)
    msms_result = CalEvaluate(labels_true, labels_pred)
    return msms_result

def MSMS2Vec(spectrums):
    m = ms_to_vec()
    peaks = [s.peaks.to_numpy for s in spectrums]
    peaks = [m.transform(i) for i in peaks]
    peaks_vec = np.array(peaks)
    return peaks_vec

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

    msbert_result = MSBERTCluster('E:/MSBERT_model/1025/MSBERT.pkl',test_ref,msms3,precursor3,smiles3)
    spec2vec_result = Spec2VecCluster('Spec2vecModel/ob_all.model',test_ref,smiles3)
    msms_result = MSMSCluster(test_ref,smiles3)
    
    
    
    
    

