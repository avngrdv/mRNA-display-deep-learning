# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 16:41:39 2021
@author: Alex Vinogradov
"""

import os
import pandas as pd
import numpy as np
import utils.Plotter as p
from config import constants
from utils.datatypes import Data, TrainingSample

def _mutagenize(pep):
    '''
    Create all saturated mutagenesis peptides for the parent peptide pep
    pep : 1D ndarray with amino acids as letter (dtype ~ <U1)
    
    out : 3D array (num_aas, num_pos, pep_len) containing all single mutants
          of the parent peptide
    '''
    
    #create a 3D array w/ shape = (num_aas, num_pos, pep_len)
    tiling_shape = (len(constants.aas), len(pep), 1)
    full_array = np.tile(pep, tiling_shape)
    
    for aa in constants.aas:
        for pos in range(pep.size):
            full_array[constants.aas.index(aa), pos, pos] = aa
            
    return full_array

def aaaa():
    return 11

def virtual_saturation_mutagenesis(pep, classifier, preprocessor, pipeline):
    '''
    For every single point mutant of the parent peptide pep, predict mutant
    fitness with a trained classifier, and plot it. 
    
    pep: 1D ndarray with amino acids as letter (dtype ~ <U1)
    classifier: an instance of Classifier object. Should contain a fully
                initialized model ready for inference.
    
    preprocessor: an instance of DataPreprocessor object. Can (should) be the
                  same object used for sequencing data preprocessing
                  
    pipeline:     an instance of Pipeline object.


    out:    None
            saves mutagenesis results to a .csv file and creates a plot using the data.
    '''
    
    str_pep_repr = ''.join(pep)
    mut = _mutagenize(pep)
    
    data = Data([
                TrainingSample(X = mut.reshape(-1, mut.shape[-1]), name=str_pep_repr) 
                ])

    pipeline.enque([
                      preprocessor.int_repr(),
                      preprocessor.featurize_X(reshape=True, repad=False),
                      classifier.predict()
                    ])

    data = pipeline.run(data)
    proba = data[str_pep_repr].proba.reshape(mut.shape[:-1])
    
    #TODO: fix saving to a predefined directory
    if not os.path.isdir('../peptide_interrogation/virtual_saturated_mutagenesis_v2'):
        os.makedirs('../peptide_interrogation/virtual_saturated_mutagenesis_v2')
        
    figname = os.path.join('../peptide_interrogation/virtual_saturated_mutagenesis_v2', f'mutagenesis_for_pep_{str_pep_repr}')
    p.virtual_mutagenesis(proba, pep, figname)

    df = pd.DataFrame(data=proba, index=constants.aas, columns=pep)
    df.to_csv(os.path.join('../peptide_interrogation/virtual_saturated_mutagenesis_v2',  f'mutagenesis_for_pep_{str_pep_repr}.csv'))

    return

















