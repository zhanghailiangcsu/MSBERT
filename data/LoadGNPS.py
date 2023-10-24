# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 09:50:08 2023

@author: Administrator
"""
import numpy as np
from collections import Counter
from tqdm import tqdm
from matchms.importing import load_from_msp
import pickle
from matchms.filtering import default_filters
from matchms.filtering import add_parent_mass, derive_adduct_from_name
from matchms.filtering import harmonize_undefined_inchikey, harmonize_undefined_inchi
from matchms.filtering import harmonize_undefined_smiles
from matchms.filtering import repair_inchi_inchikey_smiles
from matchms.filtering import derive_inchi_from_smiles, derive_smiles_from_inchi
from matchms.filtering import derive_inchikey_from_inchi
from matchms.filtering import select_by_mz
from matchms.filtering import normalize_intensities
from matchms.filtering import require_minimum_number_of_peaks

def CountAnnotations(spectra):
    '''
    
    '''
    inchi_lst = []
    smiles_lst = []
    inchikey_lst = []
    for i, spec in tqdm(enumerate(spectra)):
        inchi_lst.append(spec.get("inchi"))
        smiles_lst.append(spec.get("smiles"))
        inchikey = spec.get("inchikey")
        if inchikey is None:
            inchikey = spec.get("inchikey_inchi")
        inchikey_lst.append(inchikey)

    inchi_count = sum([1 for x in inchi_lst if x])
    smiles_count = sum([1 for x in smiles_lst if x])
    inchikey_count = sum([1 for x in inchikey_lst if x])
    print("Inchis:", inchi_count, "--", len(set(inchi_lst)), "unique")
    print("Smiles:", smiles_count, "--", len(set(smiles_lst)), "unique")
    print("Inchikeys:", inchikey_count, "--", 
          len(set([x[:14] for x in inchikey_lst if x])), "unique (first 14 characters)")

def ApplyFilters(s):
    s = default_filters(s)
    s = derive_adduct_from_name(s)
    s = add_parent_mass(s, estimate_from_adduct=True)
    return s

def CleanMetadata(s):
    s = harmonize_undefined_inchikey(s)
    s = harmonize_undefined_inchi(s)
    s = harmonize_undefined_smiles(s)
    s = repair_inchi_inchikey_smiles(s)
    return s

def CleanMetadata2(s):
    s = derive_inchi_from_smiles(s)
    s = derive_smiles_from_inchi(s)
    s = derive_inchikey_from_inchi(s)
    return s

def MinimalProcessing(spectrum):
    spectrum = normalize_intensities(spectrum)
    spectrum = select_by_mz(spectrum, mz_from=10.0, mz_to=1000.0)
    spectrum = require_minimum_number_of_peaks(spectrum, n_required=5)
    return spectrum

def CountFormulas(spectrums):
    formulas = []
    name_to_formulas = []
    for spec in tqdm(spectrums):
        if spec.get("formula"):
            formulas.append(spec.get("formula"))
            name_to_formulas.append(spec.get("compound_name") + "---" + spec.get("formula"))
    return len(formulas),len(list(set(formulas)))

def SeparateSpec(spectrums):
    spectrums_positive = []
    spectrums_negative = []
    for i, spec in enumerate(spectrums):
        if spec.get("ionmode") == "positive":
            spectrums_positive.append(spec)
        elif spec.get("ionmode") == "negative":
            spectrums_negative.append(spec)
        else:
            print(f"No ionmode found for spectrum {i} ({spec.get('ionmode')})")
    return spectrums_positive,spectrums_negative

def Annotated(spectrums_pos_processing):
    spectrums_pos_annotated = []
    for spec in tqdm(spectrums_pos_processing):
        inchikey = spec.get("inchikey")
        if inchikey is not None and len(inchikey)>13:
            if spec.get("smiles") or spec.get("inchi"):
                spectrums_pos_annotated.append(spec)
    return spectrums_pos_annotated

def PrecursorFilter(spectrums_pos_annotated):
    spectrums_filter = []
    for spec in tqdm(spectrums_pos_annotated):
        precursor = spec.metadata['precursor_mz']
        if precursor < 1000:
            spectrums_filter.append(spec)
    return spectrums_filter

def InstrumentFilter(spectrums_pos_annotated):
    orbitrap = []
    qtof = []
    other = []
    for spec in tqdm(spectrums_pos_annotated):
        instrument = spec.metadata['instrument']
        instrument = instrument.lower()
        if instrument == 'orbitrap':
            orbitrap.append(spec)
        elif 'qtof' in instrument or 'q-tof' in instrument or 'tof'in instrument:
            qtof.append(spec)
        else:
            other.append(spec)
    return orbitrap,qtof,other

def MakeDataset(spectrums_pos_annotated,n_max=100,test_size=0.2,n_decimals=2):
    smiles_unique = []
    train_ref = []
    train_query = []
    test_ref = []
    test_query = []
    for spec in tqdm(spectrums_pos_annotated):
        smi = spec.get("smiles")
        smiles_unique.append(smi)
    smiles_unique = list(set(smiles_unique))
    spectrum_all = [[] for i in range(len(smiles_unique))]
    for spec in tqdm(spectrums_pos_annotated):
        smi = spec.get("smiles")
        position = smiles_unique.index(smi)
        spectrum_all[position].append(spec)
    number = int(len(spectrum_all)*(1-test_size))
    train_ref.append(spectrum_all[0])
    for spec_list in tqdm(spectrum_all[1:number]):
        if len(spec_list)  == 1:
            train_ref.append(spec_list)
        if len(spec_list) > 1:
            p = np.random.choice(len(spec_list))
            train_query.append([spec_list[p]])
            spec_list.pop(p)
            train_ref.append(spec_list)
    for spec_list in tqdm(spectrum_all[number:]):
        if len(spec_list)  == 1:
            test_ref.append(spec_list)
        if len(spec_list) > 1:
            p = np.random.choice(len(spec_list))
            test_query.append([spec_list[p]])
            spec_list.pop(p)
            test_ref.append(spec_list)
    return train_ref,train_query,test_ref,test_query

def ProDataset(data,n_decimals,n_max):
    data = [s for s1 in data for s in s1]
    data_info = []
    for spec in tqdm(data):
        info = []
        peaks = spec.peaks.to_numpy
        if peaks.shape[0] > n_max:
            n_large = np.argsort(peaks[:,1])[::-1][0:n_max]
            n_large = sorted(n_large)
            peaks = peaks[n_large,:]
        smile = spec.get('smiles')
        precursor = spec.metadata['precursor_mz']
        mz = peaks[:,0]
        mz = [str(round(i,n_decimals)) for i in mz]
        inten = peaks[:,1]
        info = []
        info.append(smile)
        info.append(precursor)
        info.append([mz,inten])
        data_info.append(info)
    return data_info


if __name__ == '__main__':
    gnps_file = 'E:/MSBERT_data/ALL_GNPS.msp'
    spectrums = list(load_from_msp(gnps_file))
    print("number of spectra:", len(spectrums))
    CountAnnotations(spectrums) 
    
    spectrums = [ApplyFilters(s) for s in spectrums]
    number_formula,fornuma_unique = CountFormulas(spectrums)
    spectrums = [CleanMetadata(s) for s in tqdm(spectrums)]
    spectrums = [CleanMetadata2(s) for s in tqdm(spectrums)]
    CountAnnotations(spectrums) 
    
    spectrums_positive,spectrums_negative = SeparateSpec(spectrums)
    CountAnnotations(spectrums_positive)
    
    number_of_peaks = np.array([len(s.peaks) for s in spectrums_positive])
    print(f"{np.sum(number_of_peaks < 5)} spectra have < 5 peaks")
    
    
    
    spectrums_pos_processing = [MinimalProcessing(s) for s in spectrums_positive]
    spectrums_pos_processing = [s for s in spectrums_pos_processing if s is not None]
    CountAnnotations(spectrums_pos_processing)
    
    spectrums_pos_annotated = Annotated(spectrums_pos_processing)
    pickle.dump(spectrums_pos_annotated, open('GNPSdata/spectrums_pos_annotated.pickle','wb'))
    CountAnnotations(spectrums_pos_annotated)
    
    with open('GNPSdata/spectrums_pos_annotated.pickle', 'rb') as f:
        spectrums_pos_annotated = pickle.load(f)
    spectrums_filetr = PrecursorFilter(spectrums_pos_annotated)
    orbitrap,qtof,other = InstrumentFilter(spectrums_filetr)
    
    #Save by instrument type
    ob_train_ref,ob_train_query,ob_test_ref,ob_test_query =  MakeDataset(orbitrap,n_max=99,test_size=0.2,n_decimals=2)
    pickle.dump(ob_train_ref, open('GNPSdata/ob_train_ref.pickle','wb'))
    pickle.dump(ob_train_query, open('GNPSdata/ob_train_query.pickle','wb'))
    pickle.dump(ob_test_ref, open('GNPSdata/ob_test_ref.pickle','wb'))
    pickle.dump(ob_test_query, open('GNPSdata/ob_test_query.pickle','wb'))
    
    pickle.dump(qtof, open('GNPSdata/qtof.pickle','wb'))
    pickle.dump(other, open('GNPSdata/other.pickle','wb'))











