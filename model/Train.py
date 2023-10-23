# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 10:45:34 2023

@author: Administrator
"""
import os
os.chdir('E:/github/MSBERT')
import numpy as np
from model.MSBERTModel import BERT,MyDataSet
from LoadHMDB import LoadCFMHMDB
from process_data import make_train_data,ms_word,make_test_data
import torch.utils.data as Data
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from info_nce import InfoNCE
from timm.scheduler import CosineLRScheduler

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
#chongxie 
def TrainMask(model,input_ids,intensity,batch_size,epochs,lr):
    
    input_ids_train,intensity_train,input_ids_val,intensity_val = dataset_sep(input_ids,intensity,val_size = 0.1)
    dataset = MyDataSet(input_ids_train,intensity_train)
    dataloader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataset_val = MyDataSet(input_ids_val,intensity_val)
    dataloader_val = Data.DataLoader(dataset_val, batch_size=batch_size, shuffle=True)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.AdamW(model.parameters(), lr=lr,weight_decay=0.01)
    steps = epochs*len(dataloader)
    scheduler = CosineLRScheduler(optimizer, t_initial=steps, lr_min=0.1 * lr, warmup_t=int(0.1*steps), warmup_lr_init=0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    train_loss = []
    val_loss = []
    step_count = 0
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = []
        for step,(input_id,intensity_) in enumerate(dataloader):
            input_id = input_id.to(device)
            intensity_ = intensity_.to(device)
            logits_lm1,mask_token1,pool1,_,_,_ = model(input_id, intensity_)
            loss = criterion(logits_lm1.transpose(1,2), mask_token1)
            epoch_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(step_count)
            step_count += 1 
        train_loss.append(epoch_loss)
        print(str(epoch)+'epoch train_loss'+str(np.nanmean(epoch_loss)))
        
        model.eval()
        with torch.no_grad():
            val_step_loss = []
            for step,(input_id,intensity_) in enumerate(dataloader_val):
                input_id = input_id.to(device)
                intensity_ = intensity_.to(device)
                logits_lm1,mask_token1,pool1,_,_,_ = model(input_id, intensity_)
                loss = criterion(logits_lm1.transpose(1,2), mask_token1)
                val_step_loss.append(loss.item())
            val_loss.append(val_step_loss)
            print(str(epoch)+'epoch val_loss'+str(np.nanmean(val_step_loss)))
    # torch.save(model.state_dict(),'E:/MSBERT_model/1012two_stage/mask.pkl')
    return model,train_loss,val_loss

def TrainComparative(model,input_ids,intensity,batch_size,epochs,lr,temperature = 0.01):
    input_ids_train,intensity_train,input_ids_val,intensity_val = dataset_sep(input_ids,intensity,val_size = 0.1)
    dataset = MyDataSet(input_ids_train,intensity_train)
    dataloader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataset_val = MyDataSet(input_ids_val,intensity_val)
    dataloader_val = Data.DataLoader(dataset_val, batch_size=batch_size, shuffle=True)
    
    infoloss = InfoNCE(temperature=temperature)
    optimizer = optim.AdamW(model.parameters(), lr=0.1*lr,weight_decay=0.01)
    steps = epochs*len(dataloader)
    scheduler = CosineLRScheduler(optimizer, t_initial=steps, lr_min=0.1 * lr, warmup_t=int(0.1*steps), warmup_lr_init=0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    train_loss = []
    val_loss = []
    step_count = 0
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = []
        for step,(input_id,intensity_) in enumerate(dataloader):
            input_id = input_id.to(device)
            intensity_ = intensity_.to(device)
            _,_,pool1,_,_,pool2 = model(input_id, intensity_)
            loss = infoloss(pool1.squeeze(),pool2.squeeze())
            epoch_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(step_count)
            step_count += 1 
        train_loss.append(epoch_loss)
        print(str(epoch)+'epoch train_loss'+str(np.nanmean(epoch_loss)))
        
        model.eval()
        with torch.no_grad():
            val_step_loss = []
            for step,(input_id,intensity_) in enumerate(dataloader_val):
                input_id = input_id.to(device)
                intensity_ = intensity_.to(device)
                _,_,pool1,_,_,pool2 = model(input_id, intensity_)
                loss = infoloss(pool1.squeeze(),pool2.squeeze())
                val_step_loss.append(loss.item())
            val_loss.append(val_step_loss)
            print(str(epoch)+'epoch val_loss'+str(np.nanmean(val_step_loss)))
    return model,train_loss,val_loss

def TrainMSBERT(model,input_ids,intensity,batch_size,epochs,lr,temperature = 0.01):
    
    input_ids_train,intensity_train,input_ids_val,intensity_val = dataset_sep(input_ids,intensity,val_size = 0.1)
    dataset = MyDataSet(input_ids_train,intensity_train)
    dataloader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataset_val = MyDataSet(input_ids_val,intensity_val)
    dataloader_val = Data.DataLoader(dataset_val, batch_size=batch_size, shuffle=True)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    infoloss = InfoNCE(temperature=temperature)
    optimizer = optim.AdamW(model.parameters(), lr=lr,weight_decay=0.01)
    steps = epochs*len(dataloader)
    scheduler = CosineLRScheduler(optimizer, t_initial=steps, lr_min=0.1 * lr, warmup_t=int(0.1*steps), warmup_lr_init=0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    train_loss = []
    val_loss = []
    step_count = 0
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = []
        for step,(input_id,intensity_) in enumerate(dataloader):
            input_id = input_id.to(device)
            intensity_ = intensity_.to(device)
            logits_lm1,mask_token1,pool1,logits_lm2,mask_token2,pool2 = model(input_id, intensity_)
            loss1 = criterion(logits_lm1.transpose(1,2), mask_token1)
            loss2 = criterion(logits_lm2.transpose(1,2), mask_token2)
            loss3 = infoloss(pool1.squeeze(),pool2.squeeze())
            loss = loss1+loss2+loss3
            epoch_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(step_count)
            step_count += 1 
        train_loss.append(epoch_loss)
        print(str(epoch)+'epoch train_loss'+str(np.nanmean(epoch_loss)))
        
        model.eval()
        with torch.no_grad():
            val_step_loss = []
            for step,(input_id,intensity_) in enumerate(dataloader_val):
                input_id = input_id.to(device)
                intensity_ = intensity_.to(device)
                logits_lm1,mask_token1,pool1,logits_lm2,mask_token2,pool2 = model(input_id, intensity_)
                loss1 = criterion(logits_lm1.transpose(1,2), mask_token1)
                loss2 = criterion(logits_lm2.transpose(1,2), mask_token2)
                loss3 = infoloss(pool1.squeeze(),pool2.squeeze())
                loss = loss1+loss2+loss3
                val_step_loss.append(loss.item())
            val_loss.append(val_step_loss)
            print(str(epoch)+'epoch val_loss'+str(np.nanmean(val_step_loss)))
    return model,train_loss,val_loss
    
   
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

