# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 12:38:50 2024

@author: Administrator
"""

import numpy as np

from matchms.filtering import default_filters, normalize_intensities
from matchms.similarity.BaseSimilarity import BaseSimilarity
import numba
from tqdm import tqdm
import torch
from data.LoadGNPS import ProDataset
from data.ProcessData import MakeTrainData
from model.MSBERTModel import MyDataSet,MSBERT
import torch.utils.data as Data

@numba.njit
def cosine_similarity_matrix(vectors_1: np.ndarray, vectors_2: np.ndarray) -> np.ndarray:
   
    assert vectors_1.shape[1] == vectors_2.shape[1], "Input vectors must have same shape."
    vectors_1 = vectors_1.astype(np.float64)  
    vectors_2 = vectors_2.astype(np.float64)
    norm_1 = np.sqrt(np.sum(vectors_1**2, axis=1))
    norm_2 = np.sqrt(np.sum(vectors_2**2, axis=1))
    for i in range(vectors_1.shape[0]):
        vectors_1[i] = vectors_1[i] / norm_1[i]
    for i in range(vectors_2.shape[0]):
        vectors_2[i] = vectors_2[i] / norm_2[i]
    return np.dot(vectors_1, vectors_2.T)

@numba.njit
def cosine_similarity(vector1: np.ndarray, vector2: np.ndarray) -> np.float64:
    
    assert vector1.shape[0] == vector2.shape[0], "Input vector must have same shape."
    prod12 = 0
    prod11 = 0
    prod22 = 0
    for i in range(vector1.shape[0]):
        prod12 += vector1[i] * vector2[i]
        prod11 += vector1[i] * vector1[i]
        prod22 += vector2[i] * vector2[i]
    cosine_score = 0
    if prod11 != 0 and prod22 != 0:
        cosine_score = prod12 / np.sqrt(prod11 * prod22)
    return np.float64(cosine_score)

def FilterSpectra(spectra):
    spectra_filter = []
    for s in spectra:
        s = default_filters(s)
        s = normalize_intensities(s)
        spectra_filter.append(s)
    return spectra_filter


class MSBERTSimilarity(BaseSimilarity):
    
    def __init__(self,progress_bar: bool = False):
        self.MSBERTmodel = MSBERT(100002,512,6,16,0,100,2)
        self.MSBERTmodel.load_state_dict(torch.load('model/MSBERT.pkl'))
        self.vector_size = 512
        self.disable_progress_bar = not progress_bar
    
    def pair(self,reference,query):
        '''
        Parameters
        ----------
        reference : Spectrum
            Reference spectrum.
        query : Spectrum
            Query spectrum.
        
        Returns
        -------
        sim : float
            similarity score.
        '''
        reference_vector = self.CalEmbedding(reference)
        query_vector = self.CalEmbedding(query)
        sim = cosine_similarity(reference_vector, query_vector)
        return sim
    
    def matrix(self,references,queries,
               array_type: str = "numpy",
               is_symmetric: bool = False):
        '''
        Parameters
        ----------
        reference : list[Spectrum]
            List of reference spectra.
        query : list[Spectrum]
            List of query spectra.

        Returns
        -------
        similarity : np.ndarray

        '''
        n_rows = len(references)
        reference_vectors = np.empty((n_rows, self.vector_size), dtype="float")
        for index_reference, reference in enumerate(tqdm(references, desc='Calculating vectors of reference spectrums',
                                                         disable=self.disable_progress_bar)):
            reference_vectors[index_reference, 0:self.vector_size] = self.CalEmbedding(reference)

        n_cols = len(queries)
        if is_symmetric:
            assert np.all(references == queries), \
                "Expected references to be equal to queries for is_symmetric=True"
            query_vectors = reference_vectors
        else:
            query_vectors = np.empty((n_cols, self.vector_size), dtype="float")
            for index_query, query in enumerate(tqdm(queries, desc='Calculating vectors of query spectrums',
                                                     disable=self.disable_progress_bar)):
                query_vectors[index_query, 0:self.vector_size] = self.CalEmbedding(query)

        similarity_mat = cosine_similarity_matrix(reference_vectors, query_vectors)
        
        return similarity_mat
        
            
    def CalEmbedding(self,spectrum_in):
        '''
        Obtained MS/MS spectra embedding vectors by MSBERT
        '''
        pro_data = ProDataset([[spectrum_in]],2,99)
        msms = [i[2] for i in pro_data]
        precursor = [i[1] for i in pro_data]
        data,_ = MakeTrainData(msms,precursor,100)
        
        input_ids, intensity = zip(*data)
        intensity = [torch.FloatTensor(i) for i in intensity] 
        input_ids = [torch.LongTensor(i) for i in input_ids] 
        dataset = MyDataSet(input_ids,intensity)
        dataloader = Data.DataLoader(dataset, batch_size=16, shuffle=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.MSBERTmodel = self.MSBERTmodel.to(device)
        self.MSBERTmodel.eval()
        embed_list = []
        with torch.no_grad():
            for step,(input_id,intensity_) in tqdm(enumerate(dataloader)):
                input_id = input_id.to(device)
                intensity_ = intensity_.to(device)
                pool = self.MSBERTmodel.predict(input_id,intensity_)
                embed_list.append(pool.cpu().numpy())
        embed_arr = np.concatenate(embed_list)
        embed_arr = embed_arr.reshape(embed_arr.shape[0],embed_arr.shape[2])
        return embed_arr



