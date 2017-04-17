import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import model_utility as mut
import net_factory as nf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


# Parameters
learning_rate = 0.001
training_iters = 10
batch_size = 128
display_step = 1

# Network Parameters
n_input = 28 # MNIST data input (img shape: 28*28)
n_steps = 28 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
inputs = tf.placeholder(tf.float32, (None, n_steps, n_input))
query = tf.placeholder(tf.float32, (None, n_steps, n_input))
labels = tf.placeholder(tf.float32, (None, n_classes))


doc_var_list = [
            ['fcw',[2*n_hidden,n_classes]],
            ['fcb',[n_classes]]]

query_var_list = [
            ['fcw',[2*n_hidden,n_classes]],
            ['fcb',[n_classes]]]

doc_var = mut.create_var_tnorm('document',doc_var_list)
q_var = mut.create_var_tnorm('query',query_var_list)

with tf.variable_scope("document"):
    doc_net = nf.document_net(inputs, doc_var, n_hidden, n_steps)
with tf.variable_scope("query"):
    q_net,fw,bw = nf.query_net(query, query_var_list, n_hidden,n_steps)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=doc_net, labels=labels))


with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())  
    
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    batch_x = batch_x.reshape((batch_size, n_steps, n_input))
     
    doc_out = sess.run(doc_net, feed_dict={inputs: batch_x, labels: batch_y})
    out= sess.run(q_net, feed_dict={query: batch_x})
    outfw = sess.run(fw, feed_dict={query: batch_x})

    
    
