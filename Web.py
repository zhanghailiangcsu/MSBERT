# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 17:05:57 2024

@author: Administrator
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import streamlit as st
import pandas as pd
from matchms.importing import load_from_msp,load_from_mgf,load_from_mzml
import pickle
from data.LoadGNPS import ProDataset
from data.ProcessData import MakeTrainData
from model.MSBERTModel import MSBERT
import torch
from model.utils import ModelEmbed
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
device = torch.device("cpu")
MSBERTmodel = MSBERT(100002, 512, 6, 16, 0,100,2)
MSBERTmodel.load_state_dict(torch.load('model/MSBERT.pkl',map_location=torch.device('cpu')))


def ProcessMSP(file):
    '''
    Load dataset from MSP and prepare for MSBERT
    '''
    p = file.find('.')
    if file[p:] == '.msp':
        msms = list(load_from_msp(file))
    elif file[p:] == '.mgf':
        msms = list(load_from_mgf(file))
    elif file[p:] == '.mzML':
        msms = list(load_from_mzml(file))
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

def plot_data(data):
    fig, ax = plt.subplots()
    peaks = data.peaks.to_numpy
    x = peaks[:,0]
    y = peaks[:,1]
    plt.vlines(x,0,y,linewidths=3)
    plt.hlines(0,0,max(x)+10,linewidths=3)
    plt.xlabel('m/z',fontsize=15)
    plt.ylabel('Intensity',fontsize=15)
    plt.tick_params(labelsize=15)
    st.pyplot(fig)
    
def plot_smiles(smiles):
    fig, ax = plt.subplots()
    mol = Chem.MolFromSmiles(smiles)
    img = Draw.MolToImage(mol)
    st.image(img, use_column_width=True)
    

def GUI():
    
    title1,titl2 = st.columns([1,2])
    with titl2:
        st.title("MSBERT")
    
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
    if 'query_spec' not in st.session_state:
        st.session_state.query_spec = None
    if 'clicked_row_index' not in st.session_state:
        st.session_state.clicked_row_index = None
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'option' not in st.session_state:
        st.session_state.option = None
    if 'smiles_data' not in st.session_state:
        st.session_state.smiles_data = None
    
    app_mode = st.sidebar.selectbox('Select mode',['Query dataset',
                                                   'Reference dataset',
                                                   'Library matching'])
    if app_mode == 'Query dataset':
        st.subheader('Query dataset embedding')
        st.write('Query spectral dataset file(positive mode)')
        st.session_state.query_msp_file = st.file_uploader('Upload MSP file(msp,mgf,mzML)',
                                                           type=['msp','mgf','mzML'],
                                                           accept_multiple_files=False,key=1)
        col1,col2 = st.columns([1,1])
        with col1:
            if st.button('Embedding'):
                if st.session_state.query_msp_file is not None:
                    data_q = ProcessMSP(st.session_state.query_msp_file.name)
                    st.session_state.query_spec = list(load_from_msp(st.session_state.query_msp_file.name))
                    st.session_state.smiles_q = data_q[1]
                    st.session_state.embedding_q = EmbeddingMSBERT(MSBERTmodel,data_q)
                    with col2:
                        st.write("Query dataset embedding finished.")
                else: 
                    with col2:
                        st.write("No query dataset uploaded.")
            
    if app_mode == 'Reference dataset':
        st.subheader('Reference dataset embedding')
        st.write('The format of the reference dataset is optional.', 
                 'The first format is the msp format, and the second format is the pickle format.' ,
                 'If the reference dataset file (pickle) was saved in previous experiments,',
                 'it can be used directly to save time.')
        st.write('Reference spectral dataset file(positive mode)')
        st.session_state.reference_msp_file = st.file_uploader('Upload MSP file(msp,mgf,mzML)', 
                                                               type=['msp','mgf','mzML'],
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
            
    if app_mode == 'Library matching':
        st.subheader('Library matching')
        # col7,col8= st.columns([1,3])
        # with col7:
        if st.button('Library matching'):
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
                st.session_state.smiles_data = st.session_state.match_result
                st.session_state.df = {'index':list(np.arange(len(st.session_state.match_result))+1),
                                        'SMILES':st.session_state.match_result}
                st.subheader('MSBERT matching result table')
                for x in range(len(st.session_state.match_result)+1):
                    index,smiles,msms,image = st.columns([1.1,4,6,4])
                    if x == 0:
                        with index:
                            st.write("<h1 style='text-align: center;font-size: 18px;'>Index</h1>",unsafe_allow_html=True)
                        with smiles:
                            st.write("<h1 style='text-align: center;font-size: 18px;'>SMILES</h1>",unsafe_allow_html=True)
                        with msms:
                            st.write("<h1 style='text-align: center;font-size: 18px;'>MS/MS Spectral</h1>",unsafe_allow_html=True)
                        with image:
                            st.write("<h1 style='text-align: center;font-size: 18px;'>Molecular Structure</h1>",unsafe_allow_html=True)
                    else:
                        with index:
                            st.write(str(x))
                        with smiles:
                            st.write(st.session_state.match_result[x-1])
                        with msms:
                            plot_data(st.session_state.query_spec[x-1])
                        with image:
                            plot_smiles(st.session_state.match_result[x-1])
                        
            # st.table(st.session_state.df)
           
    # with col8:
    #     # st.session_state.option = st.text_input('Select a MS/MS for display')
    #     options = st.session_state.option = [str(i+1) for i in range(len(st.session_state.query_spec))]
    #     st.session_state.option= st.selectbox('Select a MS/MS for display', options)
       
    #     if st.button('Display MS/MS'):
    #         col9,col10 = st.columns([2,1])
    #         index = int(st.session_state.option)
    #         msms_data = st.session_state.query_spec[index]
    #         with col9:
    #             st.write('MS/MS Spectral')
    #             plot_data(msms_data)
    #         with col10:
    #             st.write('Molecular Structure')
    #             plot_smiles(st.session_state.smiles_data[index])
                    
if __name__ == '__main__':
    GUI()







