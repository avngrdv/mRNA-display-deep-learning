# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 17:59:16 2020
@author: Alex Vinogradov
"""

import os
import numpy as np
from config import constants
from config import TrackerConfig as dirs

def dense_morgan(r, w=True):
    '''
    Create a feature matrix dims = (number of aas, number of features)
    For each amino acid looked up in constants, create a list of Morgan
    fingerprints. Number of features = number of unique fingerprints.
    
    bit_features and info output values can be ignored in most cases
    (used primarily for mapping integrated gradient attributions to 
     the chemical structure of the underlying peptide)
    
    Parameters
    ----------
    r : int; maximum fingerprint radius
    w : flag True, if the resulting matrix should written to an .npy file

    Returns
    -------
    F : feature matrix
    bit_features : fingerprint accesion number (internal RDkit repr)
    info : list of dicts; fingerprint information (internal RDkit repr)
    '''
    from rdkit.Chem import AllChem
    from rdkit import Chem
    aas = [Chem.MolFromSmiles(x) for x in constants.aaSMILES]
    
    #construct a list of all bit features
    bit_features = []
    for aa in aas:
        fingerprints = AllChem.GetMorganFingerprint(aa, r)
        keys = list(fingerprints.GetNonzeroElements().keys())
        for k in keys:
            bit_features.append(k)
            
    bit_features = list(set(bit_features))
        
    #assemble the F matrix, encoding fingerprints as a dense bit string
    F = np.zeros((len(constants.aaSMILES), len(bit_features)))
    info = []
    for i, aa in enumerate(aas):
        fp_info = {}
        fingerprints = AllChem.GetMorganFingerprint(aa, r, bitInfo=fp_info).GetNonzeroElements()
        for f in fingerprints:
            F[i,bit_features.index(f)] = 1

        info.append(fp_info)

    if w:
        if not os.path.isdir('../feature_matrices'):
            os.makedirs('../feature_matrices')
            
        fname = 'DENSE_Morgan_F_r=' + str(r) + '.npy'
        np.save(os.path.join('../feature_matrices', fname), F)

    return F, bit_features, info

def bit_morgan(r, n, w=True):
    '''
    Deprecated. Unnecessarily bloats the data and has a high
    chance of feature collision unless is n is unreasonably large.
    
    Parameters
    ----------
    r : int; atomic radius of the fingerprint.
    n : int; bit length.
    w : a flag to write down the matrix as an .npy file; bool

    Returns
    -------
    Full Morgan fingerprint F matrix; ready for featurization.

    '''
    from rdkit.Chem import AllChem
    from rdkit import Chem
    
    #load amino acid SMILES
    aas = [Chem.MolFromSmiles(x) for x in constants.SMILES]
    
    #generate an empty F matrix and fill it
    F = np.zeros((len(aas), n))
    for i,aa in enumerate(aas):
        bit = AllChem.GetMorganFingerprintAsBitVect(aa, r, nBits=n)
        F[i] = list(bit.ToBitString())
    
    #write the file if need be
    if w:
        if not os.path.isdir(dirs.f_matrix):
            os.makedirs(dirs.f_matrix)
            
        fname = 'Bit_Morgan_F_r=' + str(r) + '.npy'
        np.save(os.path.join(dirs.f_matrix, fname), F)

    return F


def _get_SMILES_chars():
    '''
    Generate a dictionary enumerating all unicode characters
    found in the SMILES tuple.
    
    return:     a list containing each unique character in
                SMILES tuple. sorted.
                
                a dictionary, which has unique characters as
                keys and their value + 1 as values.
    '''
    ch = ''.join(x for x in constants.aaSMILES)
    ch = set(list(ch))
    ch = list(ch)
    ch.sort()
    return ch, {c: i+1 for i,c in enumerate(ch)}


def SMILES_repr_v1(w=True):
    '''
    Deprecated. In this representation, 
    SMILES characters are represented as one-hot vectors. 
    The representation results in humongous dimensions.
    '''
    
    SMILES_ch, _ = _get_SMILES_chars()
    dim = (len(SMILES_ch),)
    one_hot_SMILES_ch = np.diagflat(np.ones(dim, dtype=int))
    
    x_dim = len(constants.aaSMILES)
    y_dim = max([len(x) for x in constants.aaSMILES]) * dim[0]

    F = np.zeros((x_dim, y_dim)) - 1
    for i,ch in enumerate(constants.aaSMILES):
        ch_as_list = list(ch)
        row = np.array([one_hot_SMILES_ch[SMILES_ch.index(x)] for x in ch_as_list]).flatten()
        F[i,:row.size] = row

    if w:
        if not os.path.isdir(dirs.f_matrix):
            os.makedirs(dirs.f_matrix)
            
        fname = 'SMILES_repr_v1_F_matrix.npy'
        np.save(os.path.join(dirs.f_matrix, fname), F)

    return F
    

def SMILES_repr_v2(w=True):
    '''
    Deprecated. In this representation, each character of a SMILES 
    string is assigned a number (according to its rank that can be 
    looked up in get_SMILES_chars()). Representations are right-padded
    to the longest one. This representation is usually used together
    with embedding during model training.
    '''
    
    _, SMILES_d = _get_SMILES_chars()
    x_dim = len(constants.aaSMILES)
    y_dim = max([len(x) for x in constants.aaSMILES])
    
    F = np.zeros((x_dim, y_dim)) - 1
    for i,ch in enumerate(constants.aaSMILES):
        ch_as_list = list(ch)
        F[i,:len(ch_as_list)] = [SMILES_d[x] for x in ch_as_list]
    
    if w:
        if not os.path.isdir(dirs.f_matrix):
            os.makedirs(dirs.f_matrix)
            
        fname = 'SMILES_repr_v2_F_matrix.npy'
        np.save(os.path.join(dirs.f_matrix, fname), F)

    return F

    
def varimax_repr(w=True):
    '''
    See J. Comput. Biol. (2009) 16, 703-723
        doi: 10.1089/cmb.2008.0173
        
    Deprecated. Represent each amino as a varimax vector.
    dimensions of F: number of aas x number of varimax features
    '''
    
    varimax = (constants.varimax_1, 
               constants.varimax_2,
               constants.varimax_3, 
               constants.varimax_4,
               constants.varimax_5, 
               constants.varimax_6,
               constants.varimax_7, 
               constants.varimax_8)    
    
    x_dim = len(constants.aas)
    y_dim = len(varimax)
    F = np.zeros((x_dim, y_dim))
    for i,v in enumerate(varimax):
        for j,aa in enumerate(constants.aas):
            F[j,i] = v[aa]

    if w:
        if not os.path.isdir(dirs.f_matrix):
            os.makedirs(dirs.f_matrix)
            
        fname = 'varimax_F_matrix.npy'
        np.save(os.path.join(dirs.f_matrix, fname), F)

    return F



