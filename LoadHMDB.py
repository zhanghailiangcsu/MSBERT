

import os
os.chdir('E:/github/MSBERT')
import numpy as np
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt

def MsMat(components):
    peaks = [p for p in components if len(p) != 1]
    peaks = [i for i in peaks if i[0].isdigit() == True]
    peaks = [p.split() for p in peaks]
    peak = [list(map(float,p)) for p in peaks]
    msms = np.array(peak)
    return msms

def LoadCFMHMDB(file):
    '''
    Parameters
    ----------
    file : str
        Path of dataset file.
    Returns
    -------
    msms : list
        MSMS information of dataset.
    '''
    with open(file) as f:
        data = f.readlines()
        f.close()
    ids = [i for i,v in enumerate(data) if v[0:2] == 'ID']
    ids.append(len(data)+2)
    msinfo = []
    for i in tqdm(range(len(ids)-1)):
        start = ids[i]
        end = ids[i+1]-2
        components = data[start:end]
        id_number = int(components[0][3:])
        msms = MsMat(components)
        c = {'id_number':id_number,'MSMS':msms,'energy':components[2]}
        msinfo.append(c)
    ids = [i['id_number'] for i in msinfo]
    number = int(len(msinfo)/3)
    msms = []
    # m = ms_to_vec()
    for i in tqdm(range(number)):
        j = i*3
        if msinfo[j]['id_number'] == msinfo[j+1]['id_number'] and msinfo[j+1]['id_number'] == msinfo[j+2]['id_number']:
            energy0 = msinfo[j]['MSMS']
            energy1 = msinfo[j+1]['MSMS']
            energy2 = msinfo[j+2]['MSMS']
            if energy0[0,0]>1000 or energy1[0,0]>1000 or energy2[0,0]>1000:
                continue
            ms = {'id_number':ids[i*3],
                 'energy0':energy0,
                 'energy1':energy1,
                 'energy2':energy2}
            msms.append(ms)
    return msms

def CountFrag(file,n_decimals = 2):
    '''
    Parameters
    ----------
    file : TYPE
        DESCRIPTION.
    n_decimals : TYPE, optional
        DESCRIPTION. The default is 2.
    Returns
    -------
    frag : TYPE
        DESCRIPTION.
    '''
    with open(file) as f:
        data = f.readlines()
        f.close()
    peak = [i for i in data if i[0].isdigit() == True]
    peak = [float(i.split()[0]) for i in peak]
    peak = [round(i,n_decimals) for i in peak]
    peak = dict(Counter(peak))
    frag = sorted(peak.items(),key = lambda x: x[1],reverse= True)
    return frag


    
    
    











