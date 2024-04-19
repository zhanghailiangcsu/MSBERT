# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 10:23:23 2021

@author: Administrator
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse

class ms_to_vec:
    
    def __init__(self, precision = 1, max_mz = 1000, reverse = False):
        self.precision = precision
        self.max_mz = max_mz
        self.reverse = reverse
        
    def transform(self, data, sparse_vec = False, precursor=None):
        data = data[data[:,0] < self.max_mz,:]
        length = int(self.max_mz * 1 / self.precision)
        if not self.reverse:
            index = (data[:,0]  * 1 / self.precision).astype(int)
        else:
            data = data[data[:,0] < precursor,:]
            index = ((precursor - data[:,0])  * 1 / self.precision).astype(int)
            
        value = data[:,1] / np.nanmax(data[:,1])
        value[np.isnan(value)] = 0
        vector = np.zeros(length)
        for s in range(len(value)):
            vector[index[s]] = vector[index[s]]+value[s]
        if sparse_vec:
            return sparse.csr_matrix(vector)
        else:
            return vector
            
    def inverse_transform(self, vector, precursor=None):
        index = np.where(vector > 0)[0]
        if not self.reverse:
            mz = index * self.precision
        else:
            mz = precursor - index * self.precision
            
        value = vector[index]
        data = np.vstack((mz, value)).transpose()
        return np.sort(data, axis = 0)
    
    def save(self, save_path):
        np.savez(save_path, precision = self.precision, max_mz = self.max_mz, reverse = self.reverse)
        
    def load(self, save_path):
        saved = np.load(save_path, allow_pickle=True)
        self.precision = saved['precision']
        self.max_mz = saved['max_mz']
        self.reverse = saved['reverse']
        





