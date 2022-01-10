import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

"""
Created on Thu Jun 24 20:24:08 2021
@author: Alex Vinogradov
"""

import pandas as pd
import tensorflow as tf
from utils.processhandlers import Handler

class Classifier(Handler):
    '''
    A wrapper around keras-style functional API classifier models.
    Added mostly for convenience to streamline training/evaludation
    of supervised classifiers. A Data handler: public methods act on
    and transform Data.
    
    The class does not handle model definition: ready-made architectures
    need to be imported via config.py. Compatible model instances take
    two arguments: inp_dim and dropout.
    '''
    
    def __init__(self, *args):
        super(Classifier, self).__init__(*args)
        self._setup_model()
        
        super(Classifier, self)._on_completion()
        return
    
    def __repr__(self):
        return '<Classifier object>'

    def _setup_model(self):

        if not hasattr(self, 'model'):        
            msg = 'No model architecture was specified for the classifier.'
            self.logger.error(msg)
            raise ValueError(msg)
              
        if not hasattr(self, 'inp_dim'):            
            msg = 'Data input dimensions must be specified to setup the model; setup failed. . .'
            self.logger.error(msg)
            raise ValueError(msg)

        if not hasattr(self, 'drop'):       
            msg = 'Dropout rate was unspecified; defaulting to dropout=0. . .'
            self.logger.info(msg)
            self.drop = 0
                  
        if not hasattr(self, 'experiment_name'):        
            self.experiment_name = 'untitled'
            
        #model architectures passed to the classifier object should
        #have 2 arguments: input_dimensions and dropout
        self.model = self.model(inp_dim=self.inp_dim, drop=self.drop)
        self.model._name = self.experiment_name        
        
        msg = f'Classifier succesfully initialized <{self.model._name}> model.'
        self.logger.info(msg)
        
        fname = self.experiment_name + '_model_summary.log'
        with open(os.path.join(self.dirs.logs, fname), 'w') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))

        return
            
    def _validate_training_meta(self):
        
        params = ['lr', 'optimizer', 'loss', 'metrics', 'epochs', 'verbosity',
                  'shuffle_buffer', 'batch_size']
        
        for p in params:
            if not hasattr(self, p):
                msg = 'Cannot start training, because Classifier does not have the information about {p} metaparameter. . .'
                self.logger.error(msg)
                raise ValueError(msg)
        return
    
    #pipeline op
    def fit(self):
        '''
        Fit the model. All of the training meta parameters are specified 
        during class instantiation and are held in the config file.
        
        The op can either act on Data (needs 'train_data' and 'test_data' samples)
        or Data=None, in which case an automatic lookup of train and test datasets
        will be performed in self.dirs.ml_data (useful if data is featurized 
        using DataPreprocessor.to_h5 op)
        
        Parameters:
                None
    
        Returns:
                Data (no transformation); self.model will be compiled and 
                fitting will be performed
                
                Training logs will be written to a file (to self.dirs.logs)
                Model weights will be saved to self.dirs.model
        '''
        
        def train_model(data=None):

            #make sure all hyperparameters are set up
            self._validate_training_meta()
            
            #fetch data
            if data is None:                        
                train_set, test_set = self._load_h5_test_train_datasets()
            else:
                train_set = data['train_data']
                test_set = data['test_data']
    
            #setup callbacks
            callbacks = list()
            if hasattr(self, 'callbacks'):
                callbacks = self.callbacks
                            
            #compile the model
            self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
            
            #fit
            training_log = self.model.fit(
                                          train_set,
                                          epochs=self.epochs,
                                          validation_data=test_set,
                                          callbacks=callbacks,
                                          verbose=self.verbosity
                                         )
            
            #save model weights
            fname = os.path.join(self.dirs.model, self.experiment_name + '_fully_trained_model.h5')
            self.model.save_weights(fname)
            
            #dump the logs
            d = training_log.history
            d['epoch'] = [x+1 for x in training_log.epoch]
            df = pd.DataFrame.from_dict(d)
            fname = self.experiment_name + '_training_logs.csv'
            df.to_csv(os.path.join(self.dirs.logs, fname), sep=',', float_format='%.4f')
        
            return data
        return train_model

    #pipeline op
    def predict(self, flatten_results=True):
        '''
        For each sample in data, for each peptide in sample, predict its fitness.
        proba attribute is appended to the samples.
        
        Parameters:
                flatten_results:  True/False. if True, proba array is flattened
                                  to a vector
    
        Returns:
                Transformed Data object
        '''
        
        def predict_fitness(data):
            for sample in data:
                
                if sample.X.shape[1:] != self.inp_dim:
                    
                    #if it's just one peptide, recast it
                    if sample.X.shape == self.inp_dim:
                        sample.X = sample.X[None, :]
                    
                    else:
                        msg = f'Model dimensions ({self.inp_dim}) do not match data dimensions ({sample.X.shape[1:]})'
                        self.logger.error(msg)
                        raise ValueError(msg)
                    
                X = tf.convert_to_tensor(sample.X, dtype=tf.float32)
                proba = self.model.predict(X) #.numpy()
                
                if flatten_results:
                    proba = proba.ravel()
                
                sample.proba = proba
    
            return data
        return predict_fitness
    
    def generator_predict(self, generator, flatten_results=True):
        '''
        Predict peptide fitness from a generator. Should be used if samples
        don't fit the memory.
        
        Parameters:
                      generator:  a python generator to be iterated over;
                                  iterations should recover peptides in a
                                  representation compatible with model dims.
                                  
                flatten_results:  True/False. if True, proba array is flattened
                                  to a vector
    
        Returns:
                          proba:  predictions stored in an np.ndarray
        '''
        
        proba = self.model(generator).numpy()
        
        if flatten_results:
            proba = proba.ravel()
        
        return proba
        
    def load_model_weights(self, fname):
        '''
        Load model weights from fname. Searches for the model weights file 
        in self.dirs.model folder unless a full path is provided
        
        Parameters:
                          fname:  path to model weights. if only filename
                                  is provided, a lookup of the file in
                                  self.dirs.model will be performed.
    
        Returns:
                          None, done in place; self.model weights will be
                          updated
        '''        

        if not fname:
            msg = '<load_model_weights> expected filename as an argument. . .'
            self.logger.error(msg)
            raise ValueError(msg)
            
        if not os.path.isfile(fname):
            
            fname = os.path.join(self.dirs.model, fname)
            self.model.load_weights(fname)

        return 

    def _load_h5_test_train_datasets(self, *args):
        '''
        Assemble train_set, test_set and validation_set tensorflow datasets
        for model training. The op will use tensorflow-io to fetch .hdf5 datasets
        and prepare them for training.

        The .hdf5 files are looked up in self.data_dir.
        Training set should be named "train_data.hdf5"
        Test set should be named "test_data.hdf5"
        
        Parameters:
                          None
    
        Returns:
                          train_set, and test_set as tf.Dataset
        '''        
        
        import tensorflow_io as tfio
                
        train_fname = os.path.join(self.dirs.ml_data, 'train_data.hdf5')
        test_fname =  os.path.join(self.dirs.ml_data, 'test_data.hdf5')
        
        X_train = tfio.IODataset.from_hdf5(train_fname, dataset='/X', spec=tf.float32)
        y_train = tfio.IODataset.from_hdf5(train_fname, dataset='/y', spec=tf.float32)
        X_test = tfio.IODataset.from_hdf5(test_fname, dataset='/X', spec=tf.float32)
        y_test = tfio.IODataset.from_hdf5(test_fname, dataset='/y', spec=tf.float32)
        
        #queue training and testing datasets
        train_set = tf.data.Dataset.zip((X_train, y_train)).batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE).shuffle(self.shuffle_buffer, reshuffle_each_iteration=True)            
        test_set = tf.data.Dataset.zip((X_test, y_test)).batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        
        msg = 'Succesfully loaded hdf5 train and test datasets!'
        self.logger.info(msg)
        
        return (train_set, test_set)
    
