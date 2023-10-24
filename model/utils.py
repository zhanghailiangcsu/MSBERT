# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 16:58:02 2023

@author: Administrator
"""
import torch
from model.MSBERTModel import MyDataSet
import torch.utils.data as Data
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def dataset_sep(input_ids,intensity,val_size = 0.1):
    n = len(intensity)
    perm = np.random.permutation(n)
    n_train = int(n*(1-val_size))
    perm_train = perm[0:n_train]
    perm_val = perm[n_train:]
    input_ids_train = [input_ids[x] for x in perm_train] 
    input_ids_val = [input_ids[x] for x in perm_val] 
    intensity_train = [intensity[x] for x in perm_train]
    intensity_val = [intensity[x] for x in perm_val]
    return input_ids_train,intensity_train,input_ids_val,intensity_val

def model_embed(model,test_data,batch_size):
    input_ids, intensity = zip(*test_data)
    intensity = [torch.FloatTensor(i) for i in intensity] 
    dataset = MyDataSet(input_ids,intensity)
    dataloader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    embed_list = []
    with torch.no_grad():
        for step,(input_id,intensity_) in tqdm(enumerate(dataloader)):
            input_id = input_id.to(device)
            intensity_ = intensity_.to(device)
            pool = model.predict(input_id,intensity_)
            embed_list.append(pool.cpu().numpy())
    return embed_list

def search_top(dataset_arr,query_arr,dataset_smiles,query_smiles,batch):
    top1 = []
    top5 = []
    top10 = []
    start = 0
    n_dataset = np.linalg.norm(dataset_arr,axis=1)
    n_dataset = n_dataset.reshape(n_dataset.shape[0],1)
    while start < query_arr.shape[0]:
        end = start+batch
        q_i = query_arr[start:end,:]
        n_q = np.linalg.norm(q_i,axis =1)
        n_q = n_q.reshape(1,n_q.shape[0])
        n_q = np.repeat(n_q,n_dataset.shape[0],axis=0)
        dot = np.dot(dataset_arr,q_i.T)
        n_d = np.repeat(n_dataset,q_i.shape[0],axis=1)
        sim = dot/(n_d*n_q)
        sort = np.argsort(sim,axis = 0)
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

def plot_step_loss(train_loss,step=100):
    all_loss = [p for i in train_loss for p in i]
    step_loss = [all_loss[i:i+step] for i in range(0,len(all_loss),step)]
    step_loss = [np.nanmean(i) for i in step_loss]
    plt.plot(step_loss)
    plt.xlabel('Steps')
    plt.ylabel('Loss')

