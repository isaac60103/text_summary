import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import os

import model_utility as mut
import data_utility.batch_generator as bg

# Parameters
learning_rate = 0.001
training_iters = 10

display_step = 1

batch_size = 1 #32
sort_batch_count = 20
n_entities = 2
embed_size = 200
ctx_lstm_size = 256
question_lstm_size = 256
attention_mlp_hidden = 100

# Network Parameters
n_input = 128 # MNIST data input (img shape: 28*28)
n_steps = 500 # timesteps
qn_steps = 10 # Question timestep
n_hidden = 128 # hidden layer num of features
n_classes = 2 # MNIST total classes (0-9 digits)
n_docout = 10 
n_doc_state = 10


datapath = "/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/summary_case_example/mail"
qfilename = "/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/summary_case_example/question/q1.txt"
lfilename = "/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/summary_case_example/label/labels.txt"
files = os.listdir(datapath)
filelist = []

for f in files:
    filelist = filelist + [os.path.join(datapath,f)]
    
    
dql = bg.gnerate_dql_pairs_fromfile(filelist, qfilename, lfilename, 'w2v.pickle',128, n_steps,qn_steps)

contents = dql['contents'].reshape([1,n_steps,n_input])
question = dql['query'].reshape([1,qn_steps,n_input])
label = np.array([0, 1])
label = label.reshape([1,n_entities])


# tf Graph input
inputs = tf.placeholder(tf.float32, (None, n_steps, n_input))
query = tf.placeholder(tf.float32, (None, qn_steps, n_input))
labels = tf.placeholder(tf.float32, (None, n_entities))

doc_var_list = [
                ['d_attw',[2*ctx_lstm_size, attention_mlp_hidden]],
                ['q_attw',[2*question_lstm_size, attention_mlp_hidden]],
                ['wms',[attention_mlp_hidden,1]],
                ['w_rg',[2*ctx_lstm_size, n_entities]],
                ['w_ug',[2*question_lstm_size, n_entities]]
                ]
           
doc_var = mut.create_var_tnorm('Varibles',doc_var_list)

x = tf.unstack(inputs, n_steps, 1)
q = tf.unstack(query, qn_steps, 1)

with tf.variable_scope("query"):
    with tf.variable_scope("fw"):
        qlstm_fw_cell = rnn.BasicLSTMCell(question_lstm_size, forget_bias=1.0)
    # Backward direction cell
    with tf.variable_scope("bw"):
        qlstm_bw_cell = rnn.BasicLSTMCell(question_lstm_size, forget_bias=1.0)
    
    doc_net, fw, bw = rnn.static_bidirectional_rnn(qlstm_fw_cell, qlstm_bw_cell, q ,dtype=tf.float32)
    y_q = tf.concat([fw[-1], bw[-1]],1)
    
   
with tf.variable_scope("document"):
    with tf.variable_scope("fw"):
        lstm_fw_cell = rnn.BasicLSTMCell(ctx_lstm_size, forget_bias=1.0)
    with tf.variable_scope("bw"):
        lstm_bw_cell = rnn.BasicLSTMCell(ctx_lstm_size, forget_bias=1.0)
        
    docout, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)
    y_d = tf.concat(docout,0)
    
    #Attention layer
    wy_q = tf.matmul(y_q, doc_var['q_attw']) 
    wy_d = tf.add(tf.matmul(y_d, doc_var['d_attw']),wy_q)  
    m_t = tf.tanh(wy_d)
    
    #Attention Weight
    s_t = tf.nn.softmax(tf.transpose(tf.matmul(m_t,doc_var['wms'])))
    
    #Apply attention to context sum(a_t, h_i,t) --> output_dim = [1 , 2*ctx_lstm_size]
    r =  tf.matmul(s_t, y_d)   
    logit = tf.tanh(tf.matmul(r,doc_var['w_rg'])  +  tf.matmul(y_q,doc_var['w_ug'])) 
    output = tf.nn.softmax(logit)

loss = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(output), reduction_indices=[1]))
solver = tf.train.AdamOptimizer(1e-4).minimize(loss)


logfile = '../log/test'
tf.summary.scalar("Cross_Entropy",loss, collections=['train'])
merged_summary_train = tf.summary.merge_all('train')


with tf.Session() as sess:
    
    summary_writer = tf.summary.FileWriter(logfile, sess.graph)  
    
    sess.run(tf.global_variables_initializer())  
    
    for i in range(100):
        cost, out, _ , summary = sess.run([loss,output,solver, merged_summary_train], feed_dict={inputs: contents, query:question, labels:label})  
        print("Loss:{}".format(cost))
        summary_writer.add_summary(summary, i)
    
    
    summary_writer.close()
    sess.close()  

    
   
    
