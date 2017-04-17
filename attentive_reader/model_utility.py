import tensorflow as tf
import numpy as np

def create_var_tnorm(scope, var_list, mean = 0.0, stddev = 0.01):

    var_dict = {}
    with tf.name_scope(scope):
        for n in var_list:
           
            var_dict[n[0]] = tf.Variable(tf.truncated_normal(n[1], mean=mean, stddev=stddev),name =n[0])

    return var_dict