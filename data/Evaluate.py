# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 09:57:27 2023

@author: Administrator
"""
import numpy as np
from rdkit import Chem
from matchms.similarity import FingerprintSimilarity
from matchms.filtering.add_fingerprint import add_fingerprint
import pickle
import matplotlib.pyplot as plt
from data.LoadGNPS import ProDataset
from tqdm import tqdm
import seaborn as sns
from data.MS2Vec import ms_to_vec
from numpy.linalg import norm
from model.MSBERTModel import MSBERT
from model.utils import ModelEmbed,ParseOrbitrap
import torch 
from data.ProcessData import MakeTestData,MakeTrainData
from sklearn.manifold import TSNE

def JaccardScore(fp1,fp2):
    u_or_v = np.bitwise_or(fp1 != 0, fp2 != 0)
    u_and_v = np.bitwise_and(fp1 != 0, fp2 != 0)
    jaccard_score = 0
    if u_or_v.sum() != 0:
        jaccard_score = np.float64(u_and_v.sum()) / np.float64(u_or_v.sum())
    return jaccard_score

def Daylight(spectrum,nbits):
    if spectrum.get("smiles", None):
        smile = spectrum.get("smiles")
        mol = Chem.MolFromSmiles(smile)
    if spectrum.get("inchi", None):
        inchi = spectrum.get("inchi")
        mol = Chem.MolFromInchi(inchi)
    fingerprint = Chem.RDKFingerprint(mol, fpSize=nbits)
    return np.array(fingerprint)


def CalStructuralSim(spectrum1,spectrum2):
    '''
    Parameters
    ----------
    spec1 : list
        List containing spectrums.
    spec2 : list
        List containing spectrums.
    '''
    similarity = []
    for i in range(len(spectrum1)):
        fp_i = Daylight(spectrum1[i],2048)
        for j in range(i,len(spectrum2)):
            fp_j = Daylight(spectrum2[j],2048)
            smi = JaccardScore(fp_i,fp_j)
            similarity.append(smi) 
    return similarity

def CalStructuralSim2(spec1,spec2):
    '''
    Parameters
    ----------
    spec1 : list
        List containing spectrums.
    spec2 : list
        List containing spectrums.
    '''
    spec1 = [add_fingerprint(i,fingerprint_type="daylight", nbits=2048) for i in spec1]
    spec2 = [add_fingerprint(i,fingerprint_type="daylight", nbits=2048) for i in spec2]
    similarity_measure = FingerprintSimilarity(similarity_measure="jaccard")
    similarity = similarity_measure.matrix(spec1,spec2)
    return similarity

def CalCosineSim(data):
    data = [s for s1 in data for s in s1]
    consine_list = []
    m = ms_to_vec(1,1000)
    dataset_vec = []
    for spec in tqdm(data):
        peaks = spec.peaks.to_numpy
        dataset_vec.append(m.transform(peaks))
    for v1 in tqdm(dataset_vec):
        for v2 in dataset_vec:
            sim_cos = np.dot(v1,v2)/(norm(v1)*norm(v2))
            if sim_cos < 1:
                consine_list.append(sim_cos)
    return np.array(consine_list)

def CalEmbedSim(query_arr):
    consine_list = []
    for i in tqdm(range(query_arr.shape[0])):
        for j in range(query_arr.shape[0]):
            v1 = query_arr[i,:]
            v2 = query_arr[j,:]
            s = np.dot(v1,v2)/(norm(v1)*norm(v2))
            consine_list.append(s)
    return np.array(consine_list)

def plotSimDensity(similarity,consine_sim,embed_sim):
    sns.kdeplot(similarity,shade=True,color='#01a2d9',alpha=.7,label='Structural Similarity')
    sns.kdeplot(consine_sim,shade=True,color='#dc2624',alpha=.7,label='Cosine Similarity')
    sns.kdeplot(embed_sim,shade=True,color='g',alpha=.7,label='Embedding Similarity')
    sns.set(style='white')
    plt.tick_params(labelsize=16)
    plt.xlabel('Similarity',fontsize=16)
    plt.ylabel('Density',fontsize=16)
    plt.tight_layout()
    plt.legend(fontsize=16)
    plt.show()

#随机选取5个物质的谱进行可视化
def plotPCA(spec_vec2,l,name):
    pca1 = spec_vec2[:,0]
    pca2 = spec_vec2[:,1]
    l2 = np.cumsum(l)
    plt.scatter(pca1[0:l2[0]],pca2[0:l2[0]])
    plt.scatter(pca1[l2[0]:l2[1]],pca2[l2[0]:l2[1]])
    plt.scatter(pca1[l2[1]:l2[2]],pca2[l2[1]:l2[2]])
    plt.scatter(pca1[l2[2]:l2[3]],pca2[l2[2]:l2[3]])
    plt.scatter(pca1[l2[3]:l2[4]],pca2[l2[3]:l2[4]])
    plt.tight_layout()
    plt.tick_params(labelsize=18)
    # plt.savefig('D:/paper/MSBERT/MSBERT_20230813/figures/'+name+'.tif',dpi = 300)

def RandomSelect(train_ref,model,number=5):
    spec = [i for i in train_ref if 100 <= len(i) <= 150]
    position = np.random.randint(0,len(spec),number)
    spec_pca = [spec[i] for i in position]
    l = [len(i) for i in spec_pca]
    spec_pca2 = [i for s in spec_pca for i in s]
    tsne = TSNE(n_components=2)
    
    spec_vec = []
    m = ms_to_vec(1,1000)
    for s in spec_pca2:
        peaks = s.peaks.to_numpy
        spec_vec.append(m.transform(peaks))
    spec_vec = np.array(spec_vec)
    spec_vec2 = tsne.fit_transform(spec_vec)
    plt.figure()
    plotPCA(spec_vec2,l,'ms_dim')
    
    query= ProDataset(spec_pca,2,99)
    query_ms= [i[2] for i in query]
    precursor = [i[1] for i in query]
    train_data,word2idx = MakeTrainData(query_ms,precursor,100)
    query_ms = MakeTestData(query_ms,precursor,word2idx,100)
    query_arr = ModelEmbed(model,query_ms,16)
    query_arr2 = tsne.fit_transform(query_arr)
    plt.figure()
    plotPCA(query_arr2,l,'embed_dim')

def plotCIR(MSBERT_result,specvec_result,Consine_result,colors):
    '''
    Display compound identification results
    Parameters
    ----------
    MSBERT_result : list
        Compound identification results of MSBERT.
    specvec_result : list
        Compound identification results of Spec2Vec.
    specvec_result : list
        Compound identification results of Consine.
    colors : list
        Colors of different bars
    '''
    species = ("TOP1", "TOP5", "TOP10")
    penguin_means = {
    '5 mask 2': MSBERT_result,
    '10 mask 2': specvec_result,
    'Random mask': Consine_result}
    x = np.arange(len(species))
    width = 0.25
    multiplier = 0
    fig, ax = plt.subplots(layout='constrained')
    for attribute, measurement in penguin_means.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute,color=colors[multiplier])
        # ax.bar_label(rects, padding=3)
        multiplier += 1
    ax.set_ylabel('Values', fontsize=15)
    plt.tick_params(labelsize=15)
    ax.set_xticks(x + width, species)
    ax.legend(loc='upper left')
    ax.set_ylim(0, 1)

def PlotPos(withpos=[0.7544,0.8748,0.8918],withoutpos=[0.7473,0.8665,0.8873]):
    species = ("TOP1", "TOP5", "TOP10")
    colors = ['#F37878','#E8E46E']
    penguin_means = {
    'Without position embedding': withpos,
    'With position embedding': withoutpos}
    x = np.arange(len(species))
    width = 0.25
    multiplier = 0
    fig, ax = plt.subplots(layout='constrained')
    for attribute, measurement in penguin_means.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute,color=colors[multiplier])
        # ax.bar_label(rects, padding=3)
        multiplier += 1
    ax.set_ylabel('Values', fontsize=15)
    plt.tick_params(labelsize=15)
    ax.set_xticks(x + width, species)
    ax.legend(loc='upper left')
    ax.set_ylim(0, 1)
    
def PlotCluster(MSBERT_result,specvec_result,raw_result,colors):
    species = ("ARI", "Homogeneity", "Completeness",'v-score')
    penguin_means = {
    'MSBERT': MSBERT_result,
    'Spec2Vec': specvec_result,
    'Raw MS/MS': raw_result}
    x = np.arange(len(species))
    width = 0.2
    multiplier = 0
    fig, ax = plt.subplots(layout='constrained')
    for attribute, measurement in penguin_means.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute,color=colors[multiplier])
        # ax.bar_label(rects, padding=3)
        multiplier += 1
    ax.set_ylabel('Values', fontsize=15)
    plt.tick_params(labelsize=15)
    ax.set_xticks(x + width, species)
    ax.legend(loc='upper left')
    ax.set_ylim(0, 1)

if __name__ == '__main__':
    #计算结构相似性和原始的相似性和嵌入后的向量的相似性
    train_ref,msms1,precursor1,smiles1 = ParseOrbitrap('GNPSdata/ob_train_ref.pickle')
    train_query,msms2,precursor2,smiles2 =ParseOrbitrap('GNPSdata/ob_train_query.pickle')
    test_ref,msms3,precursor3,smiles3 = ParseOrbitrap('GNPSdata/ob_test_ref.pickle')
    test_query,msms4,precursor4,smiles4 = ParseOrbitrap('GNPSdata/ob_test_query.pickle')
    
    
    model_file = 'E:/MSBERT_model/1025/MSBERT.pkl'
    model = MSBERT(100002, 512, 6, 16, 0,100,3)
    model.load_state_dict(torch.load(model_file))
    
    train_data,word2idx = MakeTrainData(msms4,precursor4,100)
    query_msms = MakeTestData(msms4,precursor4,word2idx,100)
    query_arr= ModelEmbed(model,query_msms,64)
    
    data = [s for s1 in test_query for s in s1]
    similarity = CalStructuralSim2(data,data)
    similarity = similarity.ravel()
    consine_sim = CalCosineSim(test_query)
    embed_sim = CalEmbedSim(query_arr)
    plt.scatter(embed_sim,similarity,s=0.1)
    
    plotSimDensity(similarity,consine_sim,embed_sim)
    
    #随机选取5个物质的谱，评估嵌入的可视化
    with open('GNPSdata/ob_train_ref.pickle', 'rb') as f:
        train_ref = pickle.load(f)
    RandomSelect(train_ref,model,number=5)
    
    
    #化合物鉴定结果展示
    MSBERT_result = [0.7609,0.8748,0.8918]
    specvec_result = [0.7552,0.8508,0.8768]
    Consine_result = [0.7446,0.8443,0.8639]
    colors = ['#F37878','#E8E46E','#91D18B']
    plotCIR(MSBERT_result,specvec_result,Consine_result,colors)
    plt.savefig('D:/paper/MSBERT/MSBERT_20230813/figures/mask.tif',dpi = 300)
    
    #聚类结果展示
    MSBERT_result = [0.2009,0.7744,0.8832,0.84558]
    specvec_result = [0.1095,0.7098,0.7534,0.7383]
    raw_result = [0.0143,0.6617,0.5856,0.6089]
    colors = ['#F37878','#E8E46E','#91D18B']
    PlotCluster(MSBERT_result,specvec_result,raw_result,colors)
    plt.savefig('D:/paper/MSBERT/MSBERT_20230813/figures/1/cluster.tif',dpi = 300)
    
    
    















