# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 20:08:26 2024

@author: Administrator
"""
from matchms_workflow.Similarity import FilterSpectra,MSBERTSimilarity
from matchms.importing import load_from_msp
from matchms import calculate_scores

demofile = 'matchms_workflow/demo_msms.msp'
spectra = list(load_from_msp(demofile))
spectra = FilterSpectra(spectra)

MSBERT_Similarity = MSBERTSimilarity()
scores = calculate_scores(references=spectra,
                          queries=spectra,
                          similarity_function=MSBERT_Similarity).to_array()
