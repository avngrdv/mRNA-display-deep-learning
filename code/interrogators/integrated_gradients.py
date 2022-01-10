import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 18:05:25 2021
@author: Alex Vinogradov
"""

import tensorflow as tf

def _interpolate_tensor(baseline, peptide, alphas):
    
    alphas_x = alphas[:, tf.newaxis, tf.newaxis]  
    baseline_x = tf.expand_dims(baseline, axis=0)
    input_x = tf.expand_dims(peptide, axis=0)
    delta = input_x - baseline_x
    
    return baseline_x +  alphas_x * delta


def _compute_gradients(model, peptides):
    with tf.GradientTape() as tape:
        tape.watch(peptides)
        proba = model(peptides)
    return tape.gradient(proba, peptides)


def _integral_approximation(gradients):
    # riemann_trapezoidal
    grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
    integrated_gradients = tf.math.reduce_mean(grads, axis=0)
    return integrated_gradients


@tf.function
def integrated_gradients(model,
                         peptide,
                         baseline=None,
                         m_steps=50,
                         batch_size=2048):
    '''
    Adopted with modifications from  
    https://www.tensorflow.org/tutorials/interpretability/integrated_gradients
    
    '''
    
    #0. Infer baseline.
    if baseline is None:
        tf.zeros(shape=(peptide.shape))
      
    #1. Generate alphas.
    alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1)
    
    #Initialize TensorArray outside loop to collect gradients.    
    gradient_batches = tf.TensorArray(tf.float32, size=m_steps+1)
      
    #Iterate alphas range and batch computation for speed, memory efficiency, and scaling to larger m_steps.
    for alpha in tf.range(0, len(alphas), batch_size):
        from_ = alpha
        to = tf.minimum(from_ + batch_size, len(alphas))
        alpha_batch = alphas[from_:to]
        
        #2. Generate interpolated inputs between baseline and input.
        interpolated_path_input_batch = _interpolate_tensor(baseline=baseline,
                                                            peptide=peptide,
                                                            alphas=alpha_batch)
        
        #3. Compute gradients between model outputs and interpolated inputs.
        gradient_batch = _compute_gradients(model=model,
                                           peptides=interpolated_path_input_batch)
        
        #Write batch indices and gradients to extend TensorArray.
        gradient_batches = gradient_batches.scatter(tf.range(from_, to), gradient_batch)    
    
    #Stack path gradients together row-wise into single tensor.
    total_gradients = gradient_batches.stack()
    
    #4. Integral approximation through averaging gradients.
    avg_gradients = _integral_approximation(total_gradients)
    
    #5. Scale integrated gradients with respect to input.
    integrated_gradients = (peptide - baseline) * avg_gradients
    
    return integrated_gradients
















