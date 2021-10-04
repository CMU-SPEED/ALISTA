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

gemm_module = tf.load_op_library('../tf-gemm.so')
one =  tf.complex(1.0, 0.0)
zero = tf.complex(0.0, 0.0)

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
        B = tf.complex(B_real, B_imag)
        W_real = (np.eye(self._N, dtype=np.float32) - (np.matmul (B_real, self._A.real) - np.matmul (B_imag, self._A.imag))).astype(np.float32)
        W_imag = (-np.matmul(B_real, self._A.imag) - np.matmul(B_imag, self._A.real)).astype(np.float32)
        W = tf.complex(W_real, W_imag)

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

        y = tf.complex(y_real_, y_imag_)
        xh = tf.complex(xh_real_, xh_imag_)
        input_ = tf.complex(0.0, 0.0)

        with tf.variable_scope (self._scope, reuse=True) as vs:
            for t in range (self._T):
                B_real, B_imag, W_real, W_imag, theta_ = self.vars_in_layer [t]
                B = tf.complex(B_real, B_imag)
                W = tf.complex(W_real, W_imag)
                
                # By_real_ = tf.matmul (B_real, y_real_) - tf.matmul(B_imag, y_imag_)
                # By_imag_ = tf.matmul (B_real, y_imag_) + tf.matmul(B_imag, y_real_)
                gemm_module.gemm(B, y, input_, one, zero)
                
                # input_real = tf.matmul(W_real, xh_real_) - tf.matmul(W_imag, xh_imag_) + By_real_
                # input_imag = tf.matmul (W_real, xh_imag_) + tf.matmul(W_imag, xh_real_) + By_imag_
                gemm_module.gemm(W, xh, input_, one, one)

                xh = shrink_complex (input_, theta_)
                # xh_ = tf.concat([xh_real_, xh_imag_], 0)

                xhs_.append (xh)

        return xhs_


def shrink_complex(input_, theta_):
    """
    Soft thresholding function with input input_ and threshold theta_.
    """
    input_real, input_imag = tf.math.real(input_), tf.math.imag(input_)
    norm_eps = 1e-10
    theta_ = tf.maximum(theta_, 0.0)
    numerator = tf.maximum( tf.sqrt(tf.square(input_real) + tf.square(input_imag)) - theta_, 0.0)
    denomenator = tf.maximum(tf.sqrt(tf.square(input_real) + tf.square(input_imag)), norm_eps)
    norm_fact = tf.divide(numerator, denomenator)

    output_real = input_real * norm_fact
    output_imag = input_imag * norm_fact

    return  tf.complex(output_real, output_imag)