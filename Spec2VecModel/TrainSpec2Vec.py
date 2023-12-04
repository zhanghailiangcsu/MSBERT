# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 09:08:26 2023

@author: Administrator
"""
import pickle
import numpy as np
from tqdm import tqdm
from matchms import Spectrum
from matchms.filtering import normalize_intensities
from spec2vec import SpectrumDocument,Document
from spec2vec.model_building import train_new_word2vec_model
import matplotlib.pyplot as plt
import gensim
from Spec2VecModel.spec2wordvector import spec_to_wordvector
from spec2vec import Spec2Vec
from matchms import calculate_scores
from data.LoadGNPS import ProDataset
from model.utils import  ParseOrbitrap

def gen_reference_documents(msms,n_decimals=2):
    msms2 = [i for s in msms for i in s]
    reference_documents = [SpectrumDocument(s, n_decimals=n_decimals) for s in msms2]
    return reference_documents,msms2

def parse_msms(msms):
    msms_new = []
    for ms in tqdm(msms):
        mz = np.array([float(i[5:]) for i in ms[0]])
        intensity = ms[1]
        info = np.vstack((mz,intensity)).T
        msms_new.append(info)
    return msms_new

def peak_embed(model_file,spectrums,n_decimals=2):
    model = gensim.models.Word2Vec.load(model_file)
    spectovec = spec_to_wordvector(model=model, intensity_weighting_power=0.5,allowed_missing_percentage=20)
    word2vectors = []
    for i in range(len(spectrums)):
        spectrum_in = SpectrumDocument(spectrums[i], n_decimals=n_decimals)
        vetors,_=spectovec._calculate_embedding(spectrum_in)
        word2vectors.append(vetors)
    return word2vectors

def cal_spec2vec_top(reference_documents, query_spectrums,
              spec2vec_similarity,dataset_smiles,query_smiles,batch=500):
    top1 =[]
    top5 = []
    top10 = []
    start = 0
    while start < len(query_spectrums):
        end = start+batch
        q_i = query_spectrums[start:end]
        scores = calculate_scores(reference_documents, q_i, spec2vec_similarity)
        scores = scores.to_array()
        sort = np.argsort(scores,axis = 0)
        sort = np.flipud(sort)
        for s in range(sort.shape[1]):
            smi_q = query_smiles[(s+start)]
            smi_dataset = [dataset_smiles[i] for i in sort[0:10,s]]
            if smi_q in smi_dataset:
                top10.append(1)
            smi_dataset = [dataset_smiles[i] for i in sort[0:5,s]]
            if smi_q in smi_dataset:
                top5.append(1)
            smi_dataset = [dataset_smiles[i] for i in sort[0:1,s]]
            if smi_q in smi_dataset:
                top1.append(1)
        start += batch
    top1 = len(top1)/len(query_smiles)
    top5 = len(top5)/len(query_smiles)
    top10 = len(top10)/len(query_smiles)
    return [top1,top5,top10]

def CalSpec2VecTop(Spec2vecModel,other_ref,other_query):
    ref_documents,ref_spectrums = gen_reference_documents(other_ref,n_decimals=2)
    query_documents,query_spectrums = gen_reference_documents(other_query,n_decimals=2)
    spec2vec_similarity = Spec2Vec(model=Spec2vecModel, intensity_weighting_power=0.5,
                               allowed_missing_percentage=20)
    other_ref = ProDataset(other_ref,2,99)
    other_query = ProDataset(other_query,2,99)
    smiles1 = [i[0] for i in other_ref]
    smiles2 = [i[0] for i in other_query]
    Spec2VecOtherTop = cal_spec2vec_top(ref_documents, query_spectrums,
                  spec2vec_similarity,smiles1,smiles2,batch=1000)
    return Spec2VecOtherTop

if __name__ == '__main__':
    
    train_ref,msms1,precursor1,smiles1 = ParseOrbitrap('GNPSdata/ob_train_ref.pickle')
    train_query,msms2,precursor2,smiles2 =ParseOrbitrap('GNPSdata/ob_train_query.pickle')
    test_ref,msms3,precursor3,smiles3 = ParseOrbitrap('GNPSdata/ob_test_ref.pickle')
    test_query,msms4,precursor4,smiles4 = ParseOrbitrap('GNPSdata/ob_test_query.pickle')
    
    reference_documents,spectrums = gen_reference_documents(train_ref,n_decimals=2)
    
    
    model_file = "Spec2VecModel/ob_spec2vec.model"
    # model = gensim.models.Word2Vec.load(model_file)
    model = train_new_word2vec_model(reference_documents, iterations=10, 
                                      filename=model_file,vector_size=512,
                                      workers=10, progress_logger=True)
    query_documents,query_spectrums = gen_reference_documents(train_query,n_decimals=2)
    
    spec2vec_similarity = Spec2Vec(model=model, intensity_weighting_power=0.5,
                               allowed_missing_percentage=5.0)
    
    spec2vec_top = cal_spec2vec_top(reference_documents, query_spectrums,
                  spec2vec_similarity,smiles1,smiles2,batch=1000)
    
    
    
    
    