# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 09:39:32 2023

@author: Administrator
"""
import numpy as np
from tqdm import tqdm
import pickle
from matchms.importing import load_from_msp
from matchms.filtering import normalize_intensities
from matchms.filtering import select_by_mz
from matchms.filtering import require_minimum_number_of_peaks
from rdkit import Chem

from LoadGNPS import count_annotations,apply_filters,clean_metadata
from LoadGNPS import minimal_processing,count_formulas,separate_spec
from LoadGNPS import make_dataset,clean_metadata2,annotated

def PrecursorFilter(spectrums_pos_annotated):
    spectrums_filter = []
    for spec in tqdm(spectrums_pos_annotated):
        precursor = spec.get('precursor_mz')
        if precursor == None:
            continue
        if precursor < 1000:
            spectrums_filter.append(spec)
    return spectrums_filter

def LoadMoNAMSP(MoNA_file):
    with open(MoNA_file,encoding='utf-8') as f:
        data = f.readlines()
    dd = data[0:500]
    seg_pos = [ind for ind,v in enumerate(data) if len(v) == 1]
    seg_pos = [0]+seg_pos+[len(data)]
    spec = []
    for s in tqdm(range(len(seg_pos)-1)):
        # 现在需要的是先驱离子，mz，和仪器之类的信息，正离子模式之类的
        spec_s = data[seg_pos[s]:seg_pos[s+1]]
        if len(spec_s) > 1:
            # peak_start = 
            spec_split = [i.split(':') for i in spec_s]
            # sp = dict(spec_split)
            
            spec.append(spec_s)
    
    return spec

def LoadMassBankEUMSP(spectrums_filetr,n_decimals,n_max):
    # sp = spec_filter[0]
    for sp in tqdm(spectrums_filetr):
        smile = sp.get('smiles')
        mol = Chem.MolFromSmiles(smile)
        smile = Chem.MolToSmiles(mol)
        precursor = sp.get('precursor_mz')
        peaks = sp.peaks.to_numpy
    return peaks


if __name__ == '__main__':
    
    MoNA_file = 'E:/MSBERT_data/MoNA-export-Experimental_Spectra.msp'
    MassBankEU_file = 'E:/MSBERT_data/MassBank_NIST.msp'
    spectrums = list(load_from_msp(MassBankEU_file))
    print("number of spectra:", len(spectrums))
    count_annotations(spectrums) 
    
    spectrums = [apply_filters(s) for s in spectrums]
    number_formula,fornuma_unique = count_formulas(spectrums)
    spectrums = [clean_metadata(s) for s in tqdm(spectrums)]
    spectrums = [clean_metadata2(s) for s in tqdm(spectrums)]
    count_annotations(spectrums) 
    
    spectrums_positive,spectrums_negative = separate_spec(spectrums)
    count_annotations(spectrums_positive)
    
    number_of_peaks = np.array([len(s.peaks) for s in spectrums_positive])
    print(f"{np.sum(number_of_peaks < 5)} spectra have < 5 peaks")
    
    spectrums_pos_processing = [minimal_processing(s) for s in spectrums_positive]
    spectrums_pos_processing = [s for s in spectrums_pos_processing if s is not None]
    count_annotations(spectrums_pos_processing)
    
    spectrums_pos_annotated = annotated(spectrums_pos_processing)
    pickle.dump(spectrums_pos_annotated, open('MassBank/spectrums_pos_annotated.pickle','wb'))
    
    with open('MassBank/spectrums_pos_annotated.pickle', 'rb') as f:
        spectrums_pos_annotated = pickle.load(f)
    spectrums_filetr = PrecursorFilter(spectrums_pos_annotated)
    count_annotations(spectrums_filetr)
    
    
    smiles = [sp.get('smiles') for sp in spectrums_filetr]
    sm2 = [Chem.MolToSmiles(Chem.MolFromSmiles(sm)) for sm in smiles]
    sm3 = list(set(sm2))
    
    
    
    sp = []
    #统计一下有没有分子离子峰的比例,结果约为0.5左右
    s = spectrums_filetr[200]
    for s in spectrums_filetr:
        pre = s.metadata['precursor_mz']
        peak = s.peaks.to_numpy
        diff = min(abs(pre-peak[:,0]))
        if diff < 0.1:
            sp.append(1)
    print(len(sp)/len(spectrums_filetr))
    
    
    




















