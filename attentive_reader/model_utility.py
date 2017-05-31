import tensorflow as tf
import numpy as np

def create_var_xavier(scope, var_list, mean = 0.0, stddev = 0.01):

    var_dict = {}
    with tf.variable_scope(scope):
        for n in var_list:
           
            var_dict[n[0]] = tf.get_variable(n[0], shape=n[1], initializer=tf.contrib.layers.xavier_initializer()) 

    return var_dict