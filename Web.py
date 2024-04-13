# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 17:05:57 2024

@author: Administrator
"""

import numpy as np
import streamlit as st
import pandas as pd
from matchms.importing import load_from_msp
import pickle
from data.LoadGNPS import ProDataset
from data.ProcessData import MakeTrainData
from model.MSBERTModel import MSBERT
import torch
from model.utils import ModelEmbed
MSBERTmodel = MSBERT(100002, 512, 6, 16, 0,100,2)
MSBERTmodel.load_state_dict(torch.load('model/MSBERT.pkl'))

def ProcessMSP(file):
    '''
    Load dataset from MSP and prepare for MSBERT
    '''
    msms = list(load_from_msp(file))
    pro_data = ProDataset([msms],2,99)
    msms = [i[2] for i in pro_data]
    precursor = [i[1] for i in pro_data]
    smiles = [i[0] for i in pro_data]
    data,_ = MakeTrainData(msms,precursor,100)
    return data,smiles

def LoadPickle(filename):
    '''
    Load dataset from pickle
    '''
    with open(filename,'rb') as f:
        result = pickle.load(f)
    return result

def EmbeddingMSBERT(MSBERTmodel,data):
    data_for_model = data[0]
    arr = ModelEmbed(MSBERTmodel,data_for_model,16)
    return arr

def DatasetMatch(ref_arr,query_arr,ref_smiles,query_smiles):
    result_smiles = []
    for i in range(query_arr.shape[0]):
        q_i = query_arr[i,:]
        sim = []
        for j in range(ref_arr.shape[0]):
            r_i = ref_arr[j,:]
            sim.append(np.dot(q_i,r_i)/(np.linalg.norm(q_i)*np.linalg.norm(r_i)))
        position = np.argmax(sim)
        result_smiles.append(ref_smiles[position])
    return result_smiles

def GUI():
    st.title("MSBERT: Embedding Tandem Mass Spectra into Chemically Rational Space")
    if "embedding_q" not in st.session_state:
        st.session_state.embedding_q = 0
    if "embedding_r" not in st.session_state:
        st.session_state.embedding_r = 0
    if "smiles_q" not in st.session_state:
        st.session_state.smiles_q = None
    if "smiles_r" not in st.session_state:
        st.session_state.smiles_r = None
    if 'match_result 'not in st.session_state:
        st.session_state.match_result = None
    if "data_r" not in st.session_state:
        st.session_state.data_r = None
    if "query_msp_file" not in st.session_state:
        st.session_state.query_msp_file = None
    if "reference_msp_file" not in st.session_state:
        st.session_state.reference_msp_file = None
    if "reference_pickle_file" not in st.session_state:
        st.session_state.reference_pickle_file = None
    app_mode = st.sidebar.selectbox('Select mode',['Query dataset',
                                                   'Reference dataset',
                                                   'Dataset match'])
    if app_mode == 'Query dataset':
        st.subheader('Query dataset')
        st.write('Query spectral dataset file (.msp)')
        st.session_state.query_msp_file = st.file_uploader('Upload MSP file(.msp)', type='msp',
                                                           accept_multiple_files=False,key=1)
        col1,col2 = st.columns([1,1])
        with col1:
            if st.button('Embedding'):
                if st.session_state.query_msp_file is not None:
                    data_q = ProcessMSP(st.session_state.query_msp_file.name)
                    st.session_state.smiles_q = data_q[1]
                    st.session_state.embedding_q = EmbeddingMSBERT(MSBERTmodel,data_q)
                    with col2:
                        st.write("Query dataset embedding finished.")
                else: 
                    with col2:
                        st.write("No query dataset uploaded.")
            
    if app_mode == 'Reference dataset':
        st.subheader('Reference dataset')
        st.write('The format of the reference dataset is optional.', 
                 'The first format is the msp format, and the second format is the pickle format.' ,
                 'If the reference dataset file (pickle) was saved in previous experiments,',
                 'it can be used directly to save time.')
        st.write('Reference spectral dataset file (.msp)')
        st.session_state.reference_msp_file = st.file_uploader('Upload MSP file(.msp)', type='msp',
                                                               accept_multiple_files=False,key=3)
        st.write('Reference spectral dataset file (.pickle)')
        st.session_state.reference_pickle_file = st.file_uploader('Upload pickle file(.pickle)', type='pickle',
                                                                  accept_multiple_files=False,key=4)
        col1,col2 = st.columns([1,1])
        with col1:
            if st.button('Embedding'):
                if st.session_state.reference_msp_file is not None:
                    data_r = ProcessMSP(st.session_state.reference_msp_file.name)
                    st.session_state.smiles_r = data_r[1]
                    st.session_state.embedding_r = EmbeddingMSBERT(MSBERTmodel,data_r)
                    with col2:
                        st.write("Reference dataset embedding finished.")
                elif st.session_state.reference_pickle_file is not None:
                    info = LoadPickle(st.session_state.reference_pickle_file.name)
                    st.session_state.embedding_r = info[0]
                    st.session_state.smiles_r = info[1]
                    with col2:
                        st.write("Load embedding dataset finished.")
                else:
                    with col2:
                        st.write("No reference dataset uploaded.")
        st.write('If you want to save the embedding results of the dataset, ',
                 'you can click the save button.')         
        path = st.text_input('Save path')
        col4,col5,col6 = st.columns([1,1,1])
        with col4:
            if st.button('Save Dataset'):
                try:
                    with open(path,'wb') as f:
                        pass
                except:
                    with col5:
                        st.write('No save path')
                
                try:
                    with open(path,'wb') as f: 
                        pickle.dump([st.session_state.embedding_r,
                                     st.session_state.smiles_r],f)
                    with col5:
                        st.write('done')
                except:
                    with col6:
                        st.write('No embedding result')
            
    if app_mode == 'Dataset match':
        st.subheader('Dataset match')
        col1,col2= st.columns([1,2])
        with col1:
            if st.button('Dataset match'):
                # try:
                if type(st.session_state.embedding_r) == int:
                    with col2:
                        st.write('No reference embedding result')
                elif type(st.session_state.embedding_r) == int:
                    with col2:
                        st.write('No query embedding result')
                else:
                    st.session_state.match_result = DatasetMatch(st.session_state.embedding_r,
                                                                 st.session_state.embedding_q,
                                                                 st.session_state.smiles_r,
                                                                 st.session_state.smiles_q)
                    with col2:
                        st.write('Done')
                        df = {'index':list(np.arange(len(st.session_state.match_result))+1),
                              'SMILES':st.session_state.match_result}
                        st.dataframe(df)

if __name__ == '__main__':
    GUI()
sb2 = type(1)







