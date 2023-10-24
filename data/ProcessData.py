# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 08:50:48 2023

@author: Administrator
"""
import numpy as np

def MakeTestData(sentences,precursor,word2idx,maxlen):
    '''
    Build train test set
    '''
    intensity = [i[1] for i in sentences]
    intensity = [np.hstack((2,i)) for i in intensity]
    peaks = [i[0] for i in sentences]
    peaks = [[float(i) for i in j] for j in peaks]
    peaks = [["%.2f"%(i) for i in j] for j in peaks]
    precursor = ["%.2f"%(i) for i in precursor]
    token_list = []
    for p in range(len(peaks)):
        arr = [word2idx[s] for s in peaks[p]]
        arr = [word2idx[precursor[p]]] + arr
        token_list.append(arr)
    test_data = []
    for i in range(len(token_list)):
        input_ids = token_list[i]
        n_pad = maxlen - len(input_ids)
        input_ids.extend([word2idx['[PAD]']] * n_pad)
        intensity2 = intensity[i]/max(intensity[i])
        intensity2 = np.hstack((intensity2,np.zeros(maxlen-len(intensity2))))
        intensity2 = intensity2.reshape(1,len(intensity2))
        test_data.append([input_ids,intensity2])
    return test_data

def MakeTrainData(sentences,precursor,maxlen):
    '''
    Build train data set
    '''
    intensity = [i[1] for i in sentences]
    intensity = [np.hstack((2,i)) for i in intensity]
    peaks = [i[0] for i in sentences]
    peaks = [[float(i) for i in j] for j in peaks]
    peaks = [["%.2f"%(i) for i in j] for j in peaks]
    precursor = ["%.2f"%(i) for i in precursor]
    word_list = list(np.round(np.linspace(0,1000,100*1000,endpoint=False),2))
    word_list = ["%.2f"%(i) for i in word_list]
    word2idx = {'[PAD]' : 0, '[MASK]' : 1}
    for i, w in enumerate(word_list):
        word2idx[w] = i + 2
    token_list = []
    for p in range(len(peaks)):
        arr = [word2idx[s] for s in peaks[p]]
        arr = [word2idx[precursor[p]]] + arr
        token_list.append(arr)
    train_data = []
    for i in range(len(token_list)):
        input_ids = token_list[i]
        n_pad = maxlen - len(input_ids)
        input_ids.extend([word2idx['[PAD]']] * n_pad)
        intensity2 = intensity[i]/max(intensity[i])
        intensity2 = np.hstack((intensity2,np.zeros(maxlen-len(intensity2))))
        intensity2 = intensity2.reshape(1,len(intensity2))
        train_data.append([input_ids,intensity2])
    return train_data,word2idx









