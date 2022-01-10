# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 17:39:28 2020
@author: waxwingslain
"""

import numpy as np
from config import constants 

def from_matrix_v2(X, F=None, reshape=True, repad=False):
    '''
    A general factorizer tool, a faster matmul implementation (hence v2)
    About ~5-10x faster than the deprecated v1 solution. 
    Using cupy can further speed it up by about 10x (not implemented here).
        
        X:    P-style matrix of peptides. Should be numerically represented.
              dimensions: #peptides x sequence length  
              
        F:    Matrix containing new amino acid representations.
              each row is a new representation of an amino acid.
              Dimensions: 20 x #features to factor into
              Some F matrices may be padded if amino acids
              factorize into features of different length. Paddings 
              should the values of -1.
              
              If None, the peptides will be represented with one hot encoding
              

        repad: Should be flagged True when F is internally padded.
               Many representations, for instance one hot, have equally
               long vectors corresponding to each amino acids, but some,
               for instance, SMILES_repr_v2 are all different.
               In that case, the SMILES_repr_v2 matrix is internally padded
               to the longest representation, which upon mapping to X will
               result in pads in the middle of the sequence. Flagging repad
               will push all pads to the right
               
        reshape: Should be flagged if a peptide sequence is to be represented as
                 a 2D array. REPADDED MATRIX SHOULD NOT BE RESHAPED 
                 (it doesn't make sense but mathematically possible)
              
        out:  X-style matrix with factored representations.
              Dimensions: #peptides x sequence length * #features for each aa
    '''
    
    x_shape = X.shape
    
    X = X.ravel()
    expansion = len(constants.aas)
    
    #convert the matrix to a one-hot encoding
    fX = np.zeros((X.size, expansion))
    fX[np.arange(X.size), X] = 1
    
    #matmul by a factorization matrix to get the featurized repr
    if F is not None:   
        fX = np.matmul(fX, F)
    
    fX = np.reshape(fX, (x_shape[0], -1))
    if repad:
        #find where values of X correspond to pads (-1)
        #and then do stable argsort to move them to the right
        ind = np.argsort(fX == -1, kind='stable')
        
        #reindex the matrix
        fX = fX[np.arange(fX.shape[0])[:, None], ind]
        
        #replace -1 with 0 for output
        #truncate columns containing only pads >> risky and isnt' done
        fX[fX == -1] = 0    
    
    if reshape:
        fX = np.reshape(fX, x_shape + (-1,))
    
    return fX

def into_h5(X, y=None, path=None, F=None, reshape=False, repad=False, chunks=20):
    '''
    Featurize the matrix and write it to an hdf5 file. Mainly used when
    the featurized X doesn't fit the memory. X will be split in chunks,
    factorized chunk by chunk and written to the file.
    
        X: peptides (np 2D array, numerically represented) to be featurized
        
        y: labels as a 1D p.ndarray
        path: full path to the .hdf5 file to be created
        F, reshape, repad: featurization parameters passed to the 
                           from_matrix_v2 routine.
                           
        chunks: int; number of chunks to split the X in to. Each resulting 
                chunk should be small enough to fit the memory in the featurized
                form.
                    
        out:    None
                           
    '''

    import h5py
    
    if F is None:
        z = len(constants.aas)
    else:
        z = F.shape[-1]
    
    if reshape:
        dims = [X.shape[0], X.shape[-1], z]
    else:
        dims = [X.shape[0], X.shape[-1] * z]
        
    with h5py.File(path, 'w') as f:
        x_set = f.create_dataset("X", dims, dtype=np.float32)
        
        #n is the number of subarrays.
        indx = np.array_split(np.arange(X.shape[0]), chunks)

        for ind in indx:
            x_set[ind] = from_matrix_v2(X[ind], F=F, reshape=reshape, repad=repad)
        
        if y.ndim > 0:
            y_set = f.create_dataset("y", (y.size,), dtype=np.float32)
            y_set[...] = y   
    
    return


