# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 20:16:25 2020

@author: waxwingslain
"""

import numpy as np
import tensorflow.keras as K


def EarlyStop(monitor='val_loss', min_delta=0.00001, patience=8, verbose=1):
    return K.callbacks.EarlyStopping(monitor=monitor, min_delta=min_delta, 
                                     patience=patience, verbose=verbose)

def Checkpoint(filepath=None,
               monitor='val_loss', save_weights_only=True, verbose=1, save_best_only=False):

    return K.callbacks.ModelCheckpoint(filepath=filepath,
                                       monitor='val_loss',
                                       save_weights_only=True,
                                       verbose=1,
                                       save_best_only=False)    


class AdditionalValidationSets(K.callbacks.Callback):
    def __init__(self, validation_sets, verbose=0):

        super(AdditionalValidationSets, self).__init__()
        self.validation_sets = validation_sets
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs):

        #evaluate on the additional validation sets
        X_val, y_val = self.validation_sets
        yields = np.array([y for y in y_val.as_numpy_iterator()])
        proba = self.model.predict(X_val)[:,0]
        pcc = np.corrcoef(np.vstack((proba, yields)))   
        print('Validation set PCC:', np.round(pcc[0, 1], 4))       
        logs['PCC'] = pcc[0, 1]
                    
    
    


