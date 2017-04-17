import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import model_utility as mut


def document_net(x, varibles, n_hidden,n_steps):
   
    with tf.name_scope("document"):

        x = tf.unstack(x, n_steps, 1)

        with tf.variable_scope("fw"):
            lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
            # Backward direction cell
        with tf.variable_scope("bw"):
            lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    
        # Get lstm cell output
        
        outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                  dtype=tf.float32)
    

    #return tf.matmul(outputs[-1], varibles['fcw']) + varibles['fcb']
    return outputs

def query_net(x, varibles, n_hidden, n_steps):
    
    with tf.name_scope("query"):
        
        x = tf.unstack(x, n_steps, 1)

        with tf.variable_scope("fw"):
            qlstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
        # Backward direction cell
        with tf.variable_scope("bw"):
            qlstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    
        # Get lstm cell output
        
        outputs, fw_out, bw_out = rnn.static_bidirectional_rnn(qlstm_fw_cell, qlstm_bw_cell, x,
                                                  dtype=tf.float32)
        
        out = tf.concat([fw_out, bw_out], axis = 0)
    
    #return tf.matmul(outputs[-1], varibles['fcw']) + varibles['fcb']
    return out, fw_out,bw_out