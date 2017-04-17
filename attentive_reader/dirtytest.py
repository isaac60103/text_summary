#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 06:34:46 2017

@author: dashmoment
"""

from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
learning_rate = 0.001
training_iters = 200
batch_size = 50
display_step = 5

# Network Parameters
n_input = 28 # MNIST data input (img shape: 28*28)
n_steps = 28 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# Define weights
with tf.variable_scope("test", reuse=True):
    weights2 = {
        # Hidden layer weights => 2*n_hidden because of forward + backward cells
        'out': tf.Variable(tf.random_normal([2*n_hidden, n_classes]))
    }
    biases2 = {
        'out': tf.Variable(tf.random_normal([n_classes]))
    }


b_x = tf.unstack(x, n_steps, 1)
lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
outputs, lf, lb = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, b_x, dtype=tf.float32)
pred = tf.matmul(outputs[-1], weights2['out']) + biases2['out']
f_o = lf[-1]
lb_o = lb[-1]


init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        
        backward = sess.run(lb_o, feed_dict={x: batch_x}) #backward = tumple of [c_state, m_state]: c_state is cell state, m_state is output(hidden) state
        forward = sess.run(f_o, feed_dict={x: batch_x})
        pred = sess.run(outputs, feed_dict={x: batch_x})
        
        step = step + 1
        
        
        
        
        
        
        
        
        
        
        
        