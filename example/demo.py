# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 09:59:38 2023

@author: Administrator
"""
import pickle
import torch
import matplotlib.pyplot as plt
from data.ProcessData import MakeTrainData
from model.MSBERTModel import MSBERT
from model.utils import ModelEmbed
from scipy.spatial.distance import cosine

def PlotExample(example_msms,idx):
    plt.figure()
    plt.vlines([float(i) for i in example_msms[idx][0]],0,example_msms[idx][1])
    plt.hlines(0,0,max([float(i) for i in example_msms[idx][0]]))
    plt.xlabel('m/z')
    plt.ylabel('Intensity')
    
if __name__ == '__main__':

    with open('example/example_msms.pickle', 'rb') as f:
        example_msms = pickle.load(f)
    with open('example/example_precursor.pickle', 'rb') as f:
        example_precursor= pickle.load(f)
    
    PlotExample(example_msms,0)
    PlotExample(example_msms,1)
    example_data,word2idx= MakeTrainData(example_msms,example_precursor,100)
    
    model_file = 'E:/MSBERT_model/1025/MSBERT.pkl'
    model = MSBERT(len(word2idx), 512, 6, 16, 0,100,3)
    model.load_state_dict(torch.load(model_file))
    example_arr= ModelEmbed(model,example_data,2)
    cos = cosine(example_arr[0,:],example_arr[1,:])
