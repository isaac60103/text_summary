import tensorflow as tf
from tensorflow.contrib import rnn
import model_utility as mut

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


# Parameters
learning_rate = 0.001
training_iters = 10

display_step = 1

batch_size = 1 #32
sort_batch_count = 20
n_entities = 550
embed_size = 200
ctx_lstm_size = 256
question_lstm_size = 256
attention_mlp_hidden = 100

# Network Parameters
n_input = 28 # MNIST data input (img shape: 28*28)
n_steps = 28 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 10 # MNIST total classes (0-9 digits)
n_docout = 10 
n_doc_state = 10

# tf Graph input
inputs = tf.placeholder(tf.float32, (None, n_steps, n_input))
query = tf.placeholder(tf.float32, (None, n_steps, n_input))
labels = tf.placeholder(tf.float32, (None, n_classes))


doc_var_list = [
                ['d_attw',[2*ctx_lstm_size, attention_mlp_hidden]],
                ['q_attw',[2*question_lstm_size, attention_mlp_hidden]],
                ['wms',[attention_mlp_hidden,1]],
                ['w_rg',[2*ctx_lstm_size, n_entities]],
                ['w_ug',[2*question_lstm_size, n_entities]]
                ]
           
doc_var = mut.create_var_tnorm('document',doc_var_list)


x = tf.unstack(inputs, n_steps, 1)

with tf.variable_scope("query"):
    with tf.variable_scope("fw"):
        qlstm_fw_cell = rnn.BasicLSTMCell(question_lstm_size, forget_bias=1.0)
    # Backward direction cell
    with tf.variable_scope("bw"):
        qlstm_bw_cell = rnn.BasicLSTMCell(question_lstm_size, forget_bias=1.0)
    
    doc_net, fw, bw = rnn.static_bidirectional_rnn(qlstm_fw_cell, qlstm_bw_cell, x ,dtype=tf.float32)
    y_q = tf.concat([fw[-1], bw[-1]],1)
    
   
with tf.variable_scope("document"):
    with tf.variable_scope("fw"):
        lstm_fw_cell = rnn.BasicLSTMCell(question_lstm_size, forget_bias=1.0)
    with tf.variable_scope("bw"):
        lstm_bw_cell = rnn.BasicLSTMCell(question_lstm_size, forget_bias=1.0)
        
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
    
    output = tf.tanh(tf.matmul(r,doc_var['w_rg'])  +  tf.matmul(y_q,doc_var['w_ug'])) 


with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())  
    
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    batch_x = batch_x.reshape((batch_size, n_steps, n_input))
     

    pre = sess.run(output, feed_dict={inputs: batch_x})
    

    

    
   
    
