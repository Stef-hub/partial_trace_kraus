#!/usr/bin/env python
#-*- coding: utf-8 -*-
# projection to the nearest PSD matrix

import keras
import tensorflow as tf
from keras.engine.topology import Layer

class PSDReluLayer(Layer):

    def __init__(self, eps, **kwargs):
        self.eps = eps
        super(PSDReluLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(PSDReluLayer, self).build(input_shape)

    def call(self, x):

        s, u, v = tf.linalg.svd(x)
        a = tf.math.maximum(s, tf.multiply(self.eps, tf.ones_like(s)))
        x_relu = tf.matmul(u, tf.matmul(tf.linalg.diag(a), v, adjoint_b=True))
        return x_relu

    def compute_output_shape(self, input_shape):
        return input_shape

