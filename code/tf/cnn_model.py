# -*- coding: utf-8 -*-
"""
Created on Mon May 10 12:58:27 2021
@author: Alex Vinogradov
"""

import tensorflow.keras as K
def c1d_bn(x, 
           n_filter, 
           filter_size,
           padding='same', 
           strides=1, 
           use_bias=True,
           kernel_regularizer=None,
           dilation_rate=1):

    x = K.layers.Convolution1D(n_filter, filter_size,
                               strides=strides,
                               padding=padding,
                               use_bias=use_bias,
                               kernel_regularizer=kernel_regularizer,
                               dilation_rate=dilation_rate)(x)
    
    x = K.layers.BatchNormalization(scale=False)(x)
    x = K.layers.Activation('relu')(x)
    return x

def regular_fork(x, n_in=80, size=3, drop=0.05):
    
    branch_1 = c1d_bn(x, n_in, size, padding='same')
    branch_2 = c1d_bn(x, n_in, size-1, padding='same')
    branch_3 = c1d_bn(x, n_in, size+1, padding='same')
    x = K.layers.concatenate([branch_1, branch_2, branch_3])
    x = K.layers.Dropout(drop)(x)
               
    return x

def dilated_fork(x, n_in=80, size=3, base_rate=2, drop=0.05):
    
    branch_1 = c1d_bn(x, n_in, size, padding='same', dilation_rate=base_rate)
    branch_2 = c1d_bn(x, n_in, size, padding='same', dilation_rate=base_rate+1)
    branch_3 = c1d_bn(x, n_in, size, padding='same', dilation_rate=base_rate+2)
    x = K.layers.concatenate([branch_1, branch_2, branch_3])
    x = K.layers.Dropout(drop)(x)
               
    return x

def summary_block(x, size=256, drop=0.05):
    
    x = K.layers.Flatten()(x)
    x = K.layers.Dense(size, activation='relu', kernel_initializer='he_normal')(x)
    x = K.layers.Dropout(drop)(x)
    x = K.layers.Dense(size, activation='relu', kernel_initializer='he_normal')(x)
    x = K.layers.Dropout(drop)(x)
    out = K.layers.Dense(1, activation='sigmoid', kernel_initializer='he_normal')(x)   
    return out

def cnn_v5(inp_dim=None, drop=0.05):
 
    inp = K.layers.Input(shape=inp_dim)  
    
    x = regular_fork(inp, 
                     n_in=104, 
                     size=3,  
                     drop=drop)
    
    x = dilated_fork(x, n_in=104,
                     size=3, 
                     base_rate=2,
                     drop=drop)
    
    x = regular_fork(x, 
                     n_in=104, 
                     size=3,
                     drop=drop)
    
    x = dilated_fork(x, n_in=104,
                     size=3, 
                     base_rate=2,
                     drop=drop)
  
    out = summary_block(x, 
                        size=384, 
                        drop=drop)

    return K.Model(inputs=inp, outputs=out)
    
    
def cnn_v6(inp_dim=None, drop=0.05):
 
    inp = K.layers.Input(shape=inp_dim)  
    
    x = regular_fork(inp, 
                     n_in=256, 
                     size=3,  
                     drop=drop)
    
    x = dilated_fork(x, n_in=256,
                     size=3, 
                     base_rate=2,
                     drop=drop)
    
    x = regular_fork(x, 
                     n_in=256, 
                     size=4,
                     drop=drop)
    
    x = dilated_fork(x, n_in=256,
                     size=4, 
                     base_rate=2,
                     drop=drop)

    out = summary_block(x, 
                        size=512, 
                        drop=drop)

    return K.Model(inputs=inp, outputs=out)