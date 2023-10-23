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

def count_annotations(spectra):
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

def apply_filters(s):
    s = default_filters(s)
    s = derive_adduct_from_name(s)
    s = add_parent_mass(s, estimate_from_adduct=True)
    return s

def clean_metadata(s):
    s = harmonize_undefined_inchikey(s)
    s = harmonize_undefined_inchi(s)
    s = harmonize_undefined_smiles(s)
    s = repair_inchi_inchikey_smiles(s)
    return s

def clean_metadata2(s):
    s = derive_inchi_from_smiles(s)
    s = derive_smiles_from_inchi(s)
    s = derive_inchikey_from_inchi(s)
    return s

def minimal_processing(spectrum):
    spectrum = normalize_intensities(spectrum)
    spectrum = select_by_mz(spectrum, mz_from=10.0, mz_to=1000.0)
    spectrum = require_minimum_number_of_peaks(spectrum, n_required=5)
    return spectrum

def count_formulas(spectrums):
    formulas = []
    name_to_formulas = []
    for spec in tqdm(spectrums):
        if spec.get("formula"):
            formulas.append(spec.get("formula"))
            name_to_formulas.append(spec.get("compound_name") + "---" + spec.get("formula"))
    return len(formulas),len(list(set(formulas)))

def separate_spec(spectrums):
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

def annotated(spectrums_pos_processing):
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

def make_dataset(spectrums_pos_annotated,n_max=100,test_size=0.2,n_decimals=2):
    #首先是把smile和谱对应起来，保证训练集和测试集中没有重复的化合物，
    # 然后重复谱中随机取一个作为query
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
    # train_ref = pro_dataset(train_ref,n_decimals = n_decimals,n_max=n_max)
    # train_query = pro_dataset(train_query,n_decimals = n_decimals,n_max=n_max)
    # test_ref = pro_dataset(test_ref,n_decimals = n_decimals,n_max=n_max)
    # test_query = pro_dataset(test_query,n_decimals = n_decimals,n_max=n_max)
    return train_ref,train_query,test_ref,test_query

def pro_dataset(data,n_decimals,n_max):
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
        mz = ['peak@'+str(round(i,n_decimals)) for i in mz]
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
    pickle.dump(spectrums_pos_annotated, open('GNPSdata/spectrums_pos_annotated.pickle','wb'))
    count_annotations(spectrums_pos_annotated)
    
    with open('GNPSdata/spectrums_pos_annotated.pickle', 'rb') as f:
        spectrums_pos_annotated = pickle.load(f)
    spectrums_filetr = PrecursorFilter(spectrums_pos_annotated)
    orbitrap,qtof,other = InstrumentFilter(spectrums_filetr)
    
    train_ref,train_query,test_ref,test_query =  make_dataset(spectrums_filetr,n_max=99,test_size=0.2,n_decimals=2)
    pickle.dump(train_ref, open('GNPSdata/train_refp.pickle','wb'))
    pickle.dump(train_query, open('GNPSdata/train_queryp.pickle','wb'))
    pickle.dump(test_ref, open('GNPSdata/test_refp.pickle','wb'))
    pickle.dump(test_query, open('GNPSdata/test_queryp.pickle','wb'))
    
    
    #按照仪器类型分类存储
    ob_train_ref,ob_train_query,ob_test_ref,ob_test_query =  make_dataset(orbitrap,n_max=99,test_size=0.2,n_decimals=2)
    pickle.dump(ob_train_ref, open('GNPSdata/ob_train_ref.pickle','wb'))
    pickle.dump(ob_train_query, open('GNPSdata/ob_train_query.pickle','wb'))
    pickle.dump(ob_test_ref, open('GNPSdata/ob_test_ref.pickle','wb'))
    pickle.dump(ob_test_query, open('GNPSdata/ob_test_query.pickle','wb'))
    
    pickle.dump(qtof, open('GNPSdata/qtof.pickle','wb'))
    pickle.dump(other, open('GNPSdata/other.pickle','wb'))











