#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
file  : LISTA.py
author: Xiaohan Chen
email : chernxh@tamu.edu
last_modified : 2018-10-21

Implementation of Learned ISTA proposed by LeCun et al in 2010.
"""

import numpy as np
import tensorflow as tf
import utils.train

from utils.tf import shrink
from models.LISTA_base import LISTA_base
import os

import time
class Timer:    
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start

class LISTA_complex (LISTA_base):

    """
    Implementation of LISTA model proposed by LeCun in 2010.
    """

    def __init__(self, A, T, lam, untied, coord, scope):
        """
        :A      : Numpy ndarray. Dictionary/Sensing matrix.
        :T      : Integer. Number of layers (depth) of this LISTA model.
        :lam    : Float. The initial weight of l1 loss term in LASSO.
        :untied : Boolean. Flag of whether weights are shared within layers.
        :scope  : String. Scope name of the model.
        """
        self._A   = A.astype (np.complex)
        self._T   = T # layer
        self._lam = lam
        self._M   = self._A.shape [0]
        self._N   = self._A.shape [1]

        self._scale = 1.001 * np.linalg.norm (A, ord=2)**2
        self._theta = (self._lam / self._scale).astype(np.float32)
        if coord:
            self._theta = np.ones ((self._N, 1), dtype=np.float32) * self._theta

        self._untied = untied
        self._coord  = coord
        self._scope  = scope

        """ Set up layers."""
        self.setup_layers()


    def setup_layers(self):
        """
        Implementation of LISTA model proposed by LeCun in 2010.

        :prob: Problem setting.
        :T: Number of layers in LISTA.
        :returns:
            :layers: List of tuples ( name, xh_, var_list )
                :name: description of layers.
                :xh: estimation of sparse code at current layer.
                :var_list: list of variables to be trained seperately.

        """
        Bs_real_=[]
        Bs_imag_ = []
        Ws_real_ = []
        Ws_imag_ = []
        thetas_ = []

        # print('The dimension N is', self._N, 'M is', self._M)
        B_real = (np.transpose (self._A.real) / self._scale).astype (np.float32)
        B_imag = (np.transpose (self._A.imag) / self._scale).astype (np.float32)
        W_real = (np.eye(self._N, dtype=np.float32) - (np.matmul (B_real, self._A.real) - np.matmul (B_imag, self._A.imag))).astype(np.float32)
        W_imag = (-np.matmul(B_real, self._A.imag) - np.matmul(B_imag, self._A.real)).astype(np.float32)

        with tf.variable_scope (self._scope, reuse=False) as vs:
            # constant
            self._kA_ = tf.constant (value=self._A, dtype=tf.float32)

            Bs_real_.append (tf.get_variable (name='B_real', dtype=tf.float32,
                                         initializer=B_real))
            Bs_imag_.append(tf.get_variable(name='B_imag', dtype=tf.float32,
                                            initializer=B_imag))
            Bs_real_= Bs_real_ * self._T
            Bs_imag_ = Bs_imag_ * self._T

            if not self._untied: # tied model
                Ws_real_.append (tf.get_variable (name='W_real', dtype=tf.float32,
                                             initializer=W_real))

                Ws_imag_.append(tf.get_variable(name='W_imag', dtype=tf.float32,
                                                initializer=W_imag))
                Ws_real_ = Ws_real_ * self._T
                Ws_imag_ = Ws_imag_ * self._T

            for t in range (self._T):
                thetas_.append (tf.get_variable (name="theta_%d"%(t+1),
                                                 dtype=tf.float32,
                                                 initializer=self._theta))
                if self._untied: # untied model
                    Ws_real_.append (tf.get_variable (name="W_real%d"%(t+1),
                                                 dtype=tf.float32,
                                                 initializer=W_real))

                    Ws_imag_.append(tf.get_variable(name="W_imag%d" % (t + 1),
                                                    dtype=tf.float32,
                                                    initializer=W_imag))

        # Collection of all trainable variables in the model layer by layer.
        # We name it as `vars_in_layer` because we will use it in the manner:
        # vars_in_layer [t]
        self.vars_in_layer = list (zip (Bs_real_, Bs_imag_, Ws_real_, Ws_imag_, thetas_))

        print('Finish setup the LISTA model')

    def inference (self, y_real_, y_imag_, x0_real_ = None, x0_imag_=None):
        xhs_  = [] # collection of the regressed sparse codes


        if x0_real_ is None:
            batch_size = tf.shape (y_real_) [-1]
            xh_real_ = tf.zeros (shape=(self._N, batch_size), dtype=tf.float32)
            xh_imag_ = tf.zeros(shape=(self._N, batch_size), dtype=tf.float32)
            xh_ = tf.concat([xh_real_, xh_imag_], 0)
        else:
            xh_real_ = x0_real_
            xh_imag_ = x0_imag_

        xhs_.append (xh_)
        
        

        with tf.variable_scope (self._scope, reuse=True) as vs:
            
        
            for t in range (self._T):

                B_real, B_imag, W_real, W_imag, theta_ = self.vars_in_layer [t]

                # print(f"B_real shape is {B_real.shape}\tB_imag shape is {B_imag.shape}")
                # print(f"W_real shape is {W_real.shape}\tW_imag shape is {W_imag.shape}")
                # print(f"y_real_ shape is {y_real_.shape}\ty_imag_ shape is {y_imag_.shape}")
                # print(f"xh_real_ shape is {xh_real_.shape}\txh_imag_ shape is {xh_imag_.shape}")

                # ops computed is 8mnk
                By_real_ = tf.matmul (B_real, y_real_) - tf.matmul(B_imag, y_imag_)
                By_imag_ = tf.matmul (B_real, y_imag_) + tf.matmul(B_imag, y_real_)

                # ops computed is 8mnk 
                input_real = tf.matmul (W_real, xh_real_) - tf.matmul(W_imag, xh_imag_) + By_real_
                input_imag = tf.matmul (W_real, xh_imag_) + tf.matmul(W_imag, xh_real_) + By_imag_

                # do some normalization
                xh_real_, xh_imag_ = shrink_complex (input_real, input_imag, theta_)

                xh_ = tf.concat([xh_real_, xh_imag_], 0)
                xhs_.append (xh_)

        return xhs_


def shrink_complex(input_real, input_imag, theta_):
    """
    Soft thresholding function with input input_ and threshold theta_.
    """
    norm_eps = 1e-10
    theta_ = tf.maximum(theta_, 0.0)
    sq_in_real = tf.square(input_real)
    sq_in_imag = tf.square(input_imag)
    distance = tf.sqrt(sq_in_real + sq_in_imag)

    numer = tf.maximum(distance - theta_, 0.0)
    denom = tf.maximum(distance, norm_eps)
    norm_fact = tf.divide(numer, denom)

    # O_r = I_r * (max(sqrt(I_r^2 + I_i^2) - theta, 0)) / 
    #             (max(sqrt(I_r^2 + I_i^2), norm_eps))
    output_real = input_real * norm_fact
                           
    # O_i = I_i * (max(sqrt(I_r^2 + I_i^2) - theta, 0)) / 
    #             (max(sqrt(I_r^2 + I_i^2), norm_eps))
    output_imag = input_imag * norm_fact

    return  output_real, output_imag