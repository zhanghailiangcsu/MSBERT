# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 08:50:48 2023

@author: Administrator
"""
import random
from tqdm import tqdm
import numpy as np

def make_n_data(msms,number,n_decimals,min_len,max_mz):
    '''
    Randomly take a small number of samples for testing
    '''
    pos = np.random.permutation(range(len(msms)))[0:number]
    ms20 = []
    ms21 = []
    ms22 = []
    for n in pos:
        ms20.append(msms[n]['energy0'])
        ms21.append(msms[n]['energy1'])
        ms22.append(msms[n]['energy2'])
    ms20 = pro_msms(ms20,n_decimals,min_len,max_mz)
    ms21 = pro_msms(ms21,n_decimals,min_len,max_mz)
    ms22 = pro_msms(ms22,n_decimals,min_len,max_mz)
    return ms20,ms21,ms22

def pro_msms(msmsinfo,n_decimals,min_len,max_mz):
    mz = [i[:,0] for i in msmsinfo]
    intensity = [i[:,1] for i in msmsinfo]
    peak_info = []
    for i in tqdm(range(len(mz))):
        mz_ = [j for j in mz[i] if j <= max_mz]
        if len(mz_) < min_len:
            peak_info.append([])
            continue
        mz_ = [round(j,n_decimals) for j in mz_]
        mz_ = ['peak@'+str(j) for j in mz_]
        intensity_ = intensity[i][0:len(mz_)]
        intensity_ = intensity_*100/max(intensity_)
        peak_info.append([mz_,intensity_])
    return peak_info

def ms_word(msms,n_decimals,min_len,max_mz):
    '''目前这个函数还不够完善，等到后面需要的时候还需要加上rdkit计算分子量得到先驱离子
    再计算得到loss@xxx，最终作为输入
    '''
    msms0 = [i['energy0'] for i in msms]
    msms1 = [i['energy1'] for i in msms]
    msms2 = [i['energy2'] for i in msms]
    # msmsinfo = msms0
    msms0 =  pro_msms(msms0,n_decimals,min_len,max_mz)
    msms1 =  pro_msms(msms1,n_decimals,min_len,max_mz)
    msms2 =  pro_msms(msms2,n_decimals,min_len,max_mz)
    return msms0,msms1,msms2

# def make_train_data(sentences,maxlen,max_pred,ratio=0.15):
#     '''
#     Construct data for training
#     '''
#     peaks = [i[0] for i in sentences]
#     word_list = [x for p in peaks for x in p]
#     word_list = list(set(word_list))
#     word2idx = {'[PAD]' : 0, '[CLS]' : 1, '[SEP]' : 2, '[MASK]' : 3}
#     for i, w in enumerate(word_list):
#         word2idx[w] = i + 4
#     vocab_size = len(word2idx)
#     token_list = []
#     for p in peaks:
#         arr = [word2idx[s] for s in p]
#         token_list.append(arr)
#     train_data = []
#     for i in range(len(token_list)):
#         input_ids = [word2idx['[CLS]']] + token_list[i] + [word2idx['[SEP]']]
#         n_pred = min(max_pred, max(1, int(len(input_ids) * ratio)))
#         cand_maked_pos = [i for i, token in enumerate(input_ids) 
#                           if token != word2idx['[CLS]'] and token != word2idx['[SEP]']]
#         random.shuffle(cand_maked_pos)
        
#         masked_tokens, masked_poses = [], []
#         for pos in cand_maked_pos[:n_pred]:
#             masked_poses.append(pos)
#             masked_tokens.append(input_ids[pos])
#         if random.random() < 0.8:                         # 80%：被真实mask
#             input_ids[pos] = word2idx['[MASK]']
#         elif random.random() > 0.9:                       # 10%
#             index = random.randint(0, vocab_size - 1)     # random index in vocabulary
#             while index < 4:                       # 不能是 [PAD], [CLS], [SEP], [MASK]
#                 index = random.randint(0, vocab_size - 1)
#             input_ids[pos] = index
        
#         n_pad = maxlen - len(input_ids)
#         input_ids.extend([word2idx['[PAD]']] * n_pad)
        
#         if max_pred > n_pred:
#             n_pad = max_pred - n_pred
#             masked_tokens.extend([0] * n_pad)
#             masked_poses.extend([0] * n_pad)
#         train_data.append([input_ids, masked_tokens, masked_poses])
#     return train_data,word2idx


def make_test_data(sentences,precursor,word2idx,maxlen):
    # sentences是含有peak@xxx和intensity的列表
    intensity = [i[1] for i in sentences]
    intensity = [np.hstack((2,i)) for i in intensity]
    peaks = [i[0] for i in sentences]
    peaks = [[float(i[5:]) for i in j] for j in peaks]
    peaks = [['peak@'+"%.2f"%(i) for i in j] for j in peaks]
    precursor = ['peak@'+"%.2f"%(i) for i in precursor]
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



def make_train_data(sentences,precursor,maxlen):
    intensity = [i[1] for i in sentences]
    intensity = [np.hstack((2,i)) for i in intensity]
    peaks = [i[0] for i in sentences]
    peaks = [[float(i[5:]) for i in j] for j in peaks]
    peaks = [['peak@'+"%.2f"%(i) for i in j] for j in peaks]
    precursor = ['peak@'+"%.2f"%(i) for i in precursor]
    word_list = list(np.round(np.linspace(0,1000,100*1000,endpoint=False),2))
    word_list = ['peak@'+"%.2f"%(i) for i in word_list]
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









