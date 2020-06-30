#!/usr/bin/env python
#-*- coding: utf-8 -*-

from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
from keras.regularizers import *
from keras.initializers import *
import tensorflow as tf

class KrausLayer(Layer):

    def __init__(self, output_dim, rank, **kwargs):
        self.output_dim = output_dim
        self.rank = rank
        super(KrausLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.Ai = self.add_weight(name="Ai", shape=(self.rank, self.output_dim, input_shape[-1]), initializer='glorot_uniform', trainable=True)
        super(KrausLayer, self).build(input_shape) 

    def call(self, x):

        tmp = K.dot(self.Ai[0], x)
        tmp = K.permute_dimensions(tmp, [1,0,2])
        Y0 = K.dot(tmp, tf.transpose(self.Ai[0]))
        acc = [Y0]
        for r in range(1, self.rank):
            tmp = K.dot(self.Ai[r], x)
            tmp = K.permute_dimensions(tmp, [1,0,2])
            Yr = K.dot(tmp, tf.transpose(self.Ai[r]))
            acc.append(Yr)
        Y = tf.reduce_sum(acc, axis=0)
        return Y

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim, self.output_dim)

