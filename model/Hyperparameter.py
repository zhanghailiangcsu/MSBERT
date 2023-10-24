# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 09:02:19 2023

@author: Administrator
"""
import optuna
from model.MSBERTModel import MSBERT,MyDataSet
import torch.optim as optim
from model.utils import ParseOrbitrap,ModelEmbed,SearchTop
import torch.nn as nn
import torch
from info_nce import InfoNCE
import torch.utils.data as Data
from timm.scheduler import CosineLRScheduler
from data.LoadGNPS import MakeTrainData

def objective(trial):
    '''
    Define objective function
    '''
    torch.cuda.empty_cache()
    top1 = []
    batch_size = trial.suggest_categorical('batch_size',[16,32,64])
    n_layers = trial.suggest_categorical('n_layers',[3,4,5,6,7])
    attn_heads = trial.suggest_categorical('attn_heads',[2,4,8,16])
    epochs = trial.suggest_int('epochs',2,10)
    lr = trial.suggest_loguniform('lr',1e-5,1e-3)
    
    dataset = MyDataSet(input_ids,intensity)
    dataloader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    infoloss = InfoNCE(temperature=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MSBERT(100002, 512, n_layers, attn_heads, 0,100,3)
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr,weight_decay=0.01)
    steps = epochs*len(dataloader)
    scheduler = CosineLRScheduler(optimizer, t_initial=steps, lr_min=0.1 * lr, warmup_t=int(0.1*steps), warmup_lr_init=0)
    step_count = 0
    for epoch in range(epochs):
        model.train()
        for step,(input_id,intensity_) in enumerate(dataloader):
            input_id = input_id.to(device)
            intensity_ = intensity_.to(device)
            logits_lm1,mask_token1,pool1,logits_lm2,mask_token2,pool2 = model(input_id, intensity_)
            loss1 = criterion(logits_lm1.transpose(1,2), mask_token1)
            loss2 = criterion(logits_lm2.transpose(1,2), mask_token2)
            loss3 = infoloss(pool1.squeeze(),pool2.squeeze())
            loss = loss1+loss2+loss3
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(step_count)
            step_count += 1 
    model.eval()
    with torch.no_grad():
        train_ref_arr = ModelEmbed(model,train_ref,batch_size)
        train_query_arr = ModelEmbed(model,train_query,batch_size)
        top = SearchTop(train_ref_arr,train_query_arr,smiles1,smiles2,batch=50)
        top1 = top[0]
    return top1



if __name__ == '__main__':
    train_ref,msms1,precursor1,smiles1 = ParseOrbitrap('GNPSdata/ob_train_ref.pickle')
    train_query,msms2,precursor2,smiles2 = ParseOrbitrap('GNPSdata/ob_train_query.pickle')
    train_ref,word2idx = MakeTrainData(msms1,precursor1,100)
    train_query,word2idx = MakeTrainData(msms2,precursor2,100)
    vocab_size = len(word2idx)
    input_ids, intensity = zip(*train_ref) 
    intensity = [torch.FloatTensor(i) for i in intensity] 
    
    study_name = 'MSBERT'
    study = optuna.create_study(study_name=study_name,direction="maximize")
    study.optimize(objective, n_trials=20)
    params = study.best_params
    df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
    optuna.visualization.matplotlib.plot_optimization_history(study)
    optuna.visualization.matplotlib.plot_param_importances(study)
    optuna.visualization.matplotlib.plot_contour(study,params=['lr','epochs'])
    optuna.visualization.matplotlib.plot_slice(study,params=['lr'])
    