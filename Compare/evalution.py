# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 09:57:27 2023

@author: Administrator
"""
import os
os.chdir('E:/github/MSBERT')
import numpy as np
from rdkit import Chem
from matchms.similarity import FingerprintSimilarity
from matchms.filtering.add_fingerprint import add_fingerprint
import pickle
import matplotlib.pyplot as plt
import time
import pickle
from LoadGNPS import pro_dataset,make_dataset
from tqdm import tqdm
import seaborn as sns
from ms_vec import ms_to_vec
from numpy.linalg import norm
import pandas as pd
from MSBERTModel2 import BERT,model_embed,search_top
import torch 
from process_data import make_test_data,make_train_data
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from Train_spec2vec import gen_reference_documents,cal_spec2vec_top
import gensim
from spec2vec import Spec2Vec
from spec2vec.model_building import train_new_word2vec_model


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
    # spec = [i for i in train_ref if 5 <= len(i) <= 10]
    position = np.random.randint(0,len(spec),number)
    spec_pca = [spec[i] for i in position]
    l = [len(i) for i in spec_pca]
    spec_pca2 = [i for s in spec_pca for i in s]
    pca = PCA(n_components=number)
    tsne = TSNE(n_components=2)
    
    #原始的msms降维
    spec_vec = []
    m = ms_to_vec(1,1000)
    for s in spec_pca2:
        peaks = s.peaks.to_numpy
        spec_vec.append(m.transform(peaks))
    spec_vec = np.array(spec_vec)
    # spec_vec2 = pca.fit_transform(spec_vec)
    spec_vec2 = tsne.fit_transform(spec_vec)
    # spec_vec2 = umap_.fit_transform(spec_vec)
    plt.figure()
    plotPCA(spec_vec2,l,'ms_dim')
    
    #embed 降维
    query= pro_dataset(spec_pca,2,99)
    query_ms= [i[2] for i in query]
    precursor = [i[1] for i in query]
    train_data,word2idx = make_train_data(query_ms,precursor,100)
    query_ms = make_test_data(query_ms,precursor,word2idx,100)
    query_list = model_embed(model,query_ms,16)
    query_arr = np.concatenate(query_list)
    query_arr = query_arr.reshape(query_arr.shape[0],query_arr.shape[2])
    # query_arr2 = pca.fit_transform(query_arr)
    query_arr2 = tsne.fit_transform(query_arr)
    # query_arr2 = umap_.fit_transform(query_arr)
    plt.figure()
    plotPCA(query_arr2,l,'embed_dim')


#和spec2vec比较，比较在不同数据集下面的化合物鉴定能力
#加上有数据之后的作图

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





def spec2veconother(Spec2vecModel,qtof):
    qtof_ref,qtof_query,_,_ = make_dataset(qtof,n_max=99,test_size=0,n_decimals=2)
    ref_documents,ref_spectrums = gen_reference_documents(qtof_ref,n_decimals=2)
    query_documents,query_spectrums = gen_reference_documents(qtof_query,n_decimals=2)
    spec2vec_similarity = Spec2Vec(model=spec2vecmodel, intensity_weighting_power=0.5,
                               allowed_missing_percentage=20)
    qtof_ref = pro_dataset(qtof_ref,2,99)
    qtof_query = pro_dataset(qtof_query,2,99)
    smiles1 = [i[0] for i in qtof_ref]
    smiles2 = [i[0] for i in qtof_query]
    Spec2VecQtofTop = cal_spec2vec_top(ref_documents, query_spectrums,
                  spec2vec_similarity,smiles1,smiles2,batch=1000)
    return Spec2VecQtofTop

def CosSimOnOther(qtof_ref,qtof_query):
    qtof_ref_ = [i for s in qtof_ref for i in s]
    qtof_query_ = [i for s in qtof_query for i in s]
    qtof_ref_peak = [s.peaks.to_numpy for s in qtof_ref_]
    qtof_query_peak = [s.peaks.to_numpy for s in qtof_query_]
    m = ms_to_vec()
    qtof_ref_peak = [m.transform(i) for i in qtof_ref_peak]
    qtof_ref_peak = np.array(qtof_ref_peak)
    qtof_query_peak = [m.transform(i) for i in qtof_query_peak]
    qtof_query_peak = np.array(qtof_query_peak)
    qtof_ref = pro_dataset(qtof_ref,2,99)
    qtof_query = pro_dataset(qtof_query,2,99)
    smile_ref = [i[0] for i in qtof_ref]
    smile_query = [i[0] for i in qtof_query]
    cosinetop = search_top(qtof_ref_peak,qtof_query_peak,smile_ref,smile_query,batch=50)
    return cosinetop

def Parse_orbitrap(file):
    with open(file, 'rb') as f:
        train_ref = pickle.load(f)
    ref = pro_dataset(train_ref,2,99)
    msms = [i[2] for i in ref]
    precursor = [i[1] for i in ref]
    smiles = [i[0] for i in ref]
    return train_ref,msms,precursor,smiles

def spec2veconorbitrap(Spec2vecModel,ref,query):
    ref_documents,ref_spectrums = gen_reference_documents(ref,n_decimals=2)
    query_documents,query_spectrums = gen_reference_documents(query,n_decimals=2)
    spec2vec_similarity = Spec2Vec(model=spec2vecmodel, intensity_weighting_power=0.5,
                               allowed_missing_percentage=20)
    ref2 = pro_dataset(ref,2,99)
    query2 = pro_dataset(query,2,99)
    smiles1 = [i[0] for i in ref2]
    smiles2 = [i[0] for i in query2]
    Spec2VecQtofTop = cal_spec2vec_top(ref_documents, query_spectrums,
                  spec2vec_similarity,smiles1,smiles2,batch=1000)
    return Spec2VecQtofTop

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

if __name__ == '__main__':
    #计算结构相似性和原始的相似性和嵌入后的向量的相似性
    model_file = 'E:/MSBERT_model/912/orbitrap.pkl'
    model = BERT(100002, 512, 6, 8, 0,100,0.2,3)
    model.load_state_dict(torch.load(model_file))
    with open('GNPSdata/ob_test_query.pickle', 'rb') as f:
        test_query = pickle.load(f)
    data = [s for s1 in test_query for s in s1]
    test_query= pro_dataset(test_query,2,99)
    
    query_msms = [i[2] for i in test_query]
    precursor4 = [i[1] for i in test_query]
    train_data,word2idx = make_train_data(query_msms,precursor4,100)
    query_msms = make_test_data(query_msms,precursor4,word2idx,100)
    query_list = model_embed(model,query_msms,64)
    query_arr = np.concatenate(query_list)
    query_arr = query_arr.reshape(query_arr.shape[0],query_arr.shape[2])
    
    similarity = CalStructuralSim(data,data)
    similarity = similarity.ravel()
    consine_sim = CalCosineSim(test_query)
    embed_sim = CalEmbedSim(query_arr)
    plt.scatter(embed_sim,similarity)
    
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
    
    
    #在qtof数据上测试嵌入的效果
    
    model_file = 'E:/MSBERT_model/912/orbitrap.pkl'
    MSBERTModel = BERT(100002, 512, 6, 8, 0,100,0.2,3)
    MSBERTModel.load_state_dict(torch.load(model_file))
    
    
    
    model_file = 'spec2vec_model/ob_spec2vec_iter_10.model'
    spec2vecmodel = gensim.models.Word2Vec.load(model_file)
    Spec2VecQtofTop = spec2veconother(spec2vecmodel,qtof)
    
    
    
    
    
    qtof_ref,qtof_query,_,_ = make_dataset(qtof,n_max=99,test_size=0,n_decimals=2)
    ConsineTop = CossimOnOther(qtof_ref,qtof_query)
    
    #在其他数据上进行测试
    with open('GNPSdata/other.pickle', 'rb') as f:
        other = pickle.load(f)
    ref_data,query_data,smile_ref,smile_query = ParseOtherData(other)
    MSBERTQtofTop = bertonother(MSBERTModel,ref_data,query_data,smile_ref,smile_query)
    Spec2VecQtofTop = spec2veconother(spec2vecmodel,other)
    other_ref,other_query,_,_ = make_dataset(other,n_max=99,test_size=0,n_decimals=2)
    ConsineTop = CossimOnOther(other_ref,other_query)
    
    #在orbitrap测试结果（也就是训练的仪器类型），模型已经加载好
    #代码需要整理成函数，方便一点
    #这一部分是msbert的结果
    
    train_ref,msms1,precursor1,smiles1 = Parse_orbitrap('GNPSdata/ob_train_ref.pickle')
    train_query,msms2,precursor2,smiles2 = Parse_orbitrap('GNPSdata/ob_train_query.pickle')
    test_ref,msms3,precursor3,smiles3 = Parse_orbitrap('GNPSdata/ob_test_ref.pickle')
    test_query,msms4,precursor4,smiles4 = Parse_orbitrap('GNPSdata/ob_test_query.pickle')
    
    
    train_ref,word2idx = make_train_data(msms1,precursor1,100)
    train_query,word2idx = make_train_data(msms2,precursor2,100)
    test_ref,word2idx = make_train_data(msms3,precursor3,100)
    test_query,word2idx = make_train_data(msms4,precursor4,100)
    

    train_ref_arr = model_embed(MSBERTModel,train_ref)
    train_query_arr = model_embed(MSBERTModel,train_query)
    top = search_top(train_ref_arr,train_query_arr,smiles1,smiles2,batch=50)
    
    test_ref_arr = model_embed(MSBERTModel,test_ref)
    test_query_arr = model_embed(MSBERTModel,test_query)
    dataset_arr = np.vstack((train_ref_arr,test_ref_arr))
    smiles_list = smiles1+smiles3
    top2 = search_top(dataset_arr,test_query_arr,smiles_list,smiles4,batch=50)
    
    # train_ref_arr = MSBERT_orbitrap_Embed(model,train_ref)
    # train_query_arr = MSBERT_orbitrap_Embed(model,train_query)
    # top = search_top(train_ref_arr,train_query_arr,smiles1,smiles2,batch=50)
    
    # train_ref_arr = MSBERT_orbitrap_Embed(model,train_ref)
    # test_ref_arr = MSBERT_orbitrap_Embed(model,test_ref)
    # test_query_arr = MSBERT_orbitrap_Embed(model,test_query)
    # dataset_arr = np.vstack((train_ref_arr,test_ref_arr))
    # smiles_list = smiles1+smiles3
    # top2 = search_top(dataset_arr,test_query_arr,smiles_list,smiles4,batch=50)
    
    #这一部分的结果是consine
    ConsineTop = CossimOnOther(train_ref,train_query)
    dataset = train_ref+test_ref
    ConsineTop2 = CossimOnOther(dataset,test_query)
    #这一部分的结果是spec2vec的
    #spec2vec必须在对应的数据上训练，不然的话结果很低,
    #就算是忽略比例达到20%，top也很低
    Spec2VecObTop1 = spec2veconorbitrap(spec2vecmodel,train_ref,train_query)
    Spec2VecObTop2 = spec2veconorbitrap(spec2vecmodel,dataset,test_query)
    #只能在不同的数据上训练spec2vec，然后再比较咯
    #下面的代码就是训练加计算top的
    model_file = 'spec2vec_model/retrain/ob_all.model'
    dataset = train_ref+test_ref
    reference_documents,spectrums = gen_reference_documents(dataset,n_decimals=2)
    query_documents,query_spectrums = gen_reference_documents(test_query,n_decimals=2)
    model = train_new_word2vec_model(reference_documents, iterations=10, 
                                      filename=model_file,vector_size=512,
                                      workers=10, progress_logger=True)
    spec2vec_similarity = Spec2Vec(model=model, intensity_weighting_power=0.5,
                               allowed_missing_percentage=20.0)
    smiles_ref = smiles1+smiles3
    smiles_query = smiles4
    spec2vec_top = cal_spec2vec_top(reference_documents, query_spectrums,
                  spec2vec_similarity,smiles_ref,smiles_query,batch=1000)
    
    model_file = 'spec2vec_model/retrain/qtof.model'
    with open('GNPSdata/qtof.pickle', 'rb') as f:
        qtof = pickle.load(f)
    qtof_ref,qtof_query,_,_ = make_dataset(qtof,n_max=99,test_size=0,n_decimals=2)
    ref_documents,ref_spectrums = gen_reference_documents(qtof_ref,n_decimals=2)
    query_documents,query_spectrums = gen_reference_documents(qtof_query,n_decimals=2)
    model = train_new_word2vec_model(ref_documents, iterations=10, 
                                      filename=model_file,vector_size=512,
                                      workers=10, progress_logger=True)
    qtof_ref = pro_dataset(qtof_ref,2,99)
    qtof_query = pro_dataset(qtof_query,2,99)
    smiles1 = [i[0] for i in qtof_ref]
    smiles2 = [i[0] for i in qtof_query]
    spec2vec_similarity = Spec2Vec(model=model, intensity_weighting_power=0.5,
                               allowed_missing_percentage=20.0)
    spec2vec_top = cal_spec2vec_top(ref_documents, query_spectrums,
                  spec2vec_similarity,smiles1,smiles2,batch=1000)
    
    model_file = 'spec2vec_model/retrain/other.model'
    model = gensim.models.Word2Vec.load(model_file)
    with open('GNPSdata/other.pickle', 'rb') as f:
        other = pickle.load(f)
    other_ref,other_query,_,_ = make_dataset(other,n_max=99,test_size=0,n_decimals=2)
    ref_documents,ref_spectrums = gen_reference_documents(other_ref,n_decimals=2)
    query_documents,query_spectrums = gen_reference_documents(other_query,n_decimals=2)
    model = train_new_word2vec_model(ref_documents, iterations=10, 
                                      filename=model_file,vector_size=512,
                                      workers=10, progress_logger=True)
    other_ref = pro_dataset(other_ref,2,99)
    other_query = pro_dataset(other_query,2,99)
    smiles1 = [i[0] for i in other_ref]
    smiles2 = [i[0] for i in other_query]
    spec2vec_similarity = Spec2Vec(model=model, intensity_weighting_power=0.5,
                               allowed_missing_percentage=20.0)
    spec2vec_top2 = cal_spec2vec_top(ref_documents, query_spectrums,
                  spec2vec_similarity,smiles1,smiles2,batch=1000)















