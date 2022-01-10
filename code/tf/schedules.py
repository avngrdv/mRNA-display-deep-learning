# -*- coding: utf-8 -*-
"""
Created on Sat May 22 20:39:20 2021

@author: Alex Vinogradov
"""

import tensorflow as tf

class NoamSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(NoamSchedule, self).__init__()

    self.d_model = tf.cast(d_model, tf.float32).numpy()
    self.warmup_steps = tf.cast(warmup_steps, tf.float32).numpy()

  def __call__(self, step):
           
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
    
  def get_config(self):
      
     config = {
               'd_model': self.d_model,
               'warmup_steps': self.warmup_steps,
              }

     return config