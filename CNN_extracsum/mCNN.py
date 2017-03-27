#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 17:13:40 2017

@author: ubuntu
"""

import tensorflow as tf
import numpy as np

class mCNN(object):
    def __init__(self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0)