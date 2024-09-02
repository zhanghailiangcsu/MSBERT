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
from model.utils import ModelEmbed, ProcessMSP,MSBERTSimilarity

def PlotExample(example_msms,idx):
    plt.figure()
    plt.vlines([float(i) for i in example_msms[idx][0]],0,example_msms[idx][1])
    plt.hlines(0,0,max([float(i) for i in example_msms[idx][0]]))
    plt.xlabel('m/z')
    plt.ylabel('Intensity')
    
if __name__ == '__main__':
    
    model_file = 'model/MSBERT.pkl'
    model = MSBERT(100002, 512, 6, 16, 0,100,3)
    model.load_state_dict(torch.load(model_file))
    
    demo_file = 'example/demo_msms.msp'
    demo_data,demo_smiles = ProcessMSP(demo_file)
    demo_arr = ModelEmbed(model,demo_data,16)
    
    
    PlotExample(demo_data,0)
    PlotExample(demo_data,1)
    
    cos = MSBERTSimilarity(demo_arr,demo_arr)
