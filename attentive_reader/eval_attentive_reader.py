import tensorflow as tf
from tensorflow.contrib import rnn
import os
import pickle
import numpy as np

import model_utility as mut
import data_utility.batch_generator as bg
import data_utility.attentivereader_utility as au

checkpoint_dir = '/home/ubuntu/workspace/model/attr_vanilla_model_SGD'
checkpoint_filename = os.path.join(checkpoint_dir, 'attr_vanilla_cat.ckpt')
logfile = '/home/ubuntu/workspace/log/attr_vanilla_model_SGD'

data_config = {}
test_config = {}
model_config = {}
dql_config = {}

data_config['srcpath'] = "/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/summary_case_example/result"
data_config['label_dict'] = 'model_dict.pickle'
data_config['label_type'] = 'model'
data_config['questionfile'] = "/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/summary_case_example/question/q1/q1.txt"
data_config['data'] = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/summary_case_parsed/train2/data'
data_config['question'] = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/summary_case_parsed/train2/question'
data_config['OS'] = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/summary_case_parsed/train2/OS'
data_config['model'] = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/summary_case_parsed/train2/model'
data_config['category'] = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/summary_case_parsed/train2/category'

test_config['questionfile'] = "/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/summary_case_example/question/q1/q1.txt"
test_config['srcpath'] = "/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/summary_case_example/result"
test_config['data'] = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/summary_case_parsed/test2/data'
test_config['question'] = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/summary_case_parsed/test2/question'
test_config['OS'] = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/summary_case_parsed/test2/OS'
test_config['model'] = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/summary_case_parsed/test2/model'
test_config['category'] = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/summary_case_parsed/test2/category'

model_config['input_dim'] = 1024
model_config['doc_time_step'] = 500
model_config['query_time_step'] = 10
model_config['ctx_lstm_size'] = 256
model_config['question_lstm_size'] = 256
model_config['attention_mlp_hidden'] = 100
model_config['batch_size'] = len(os.listdir(test_config['data']))//3 
model_config['word2vec'] = 'w2v.pickle'



continue_training = 1
epoch_n = 0
Nepoch = 1
save_epoch = 300
test_epoch = 500


au.preparetraning(data_config, test_config,model_config)


with open(data_config['label_dict'], 'rb') as handle:
       	label_dict = pickle.load(handle)

model_config['n_entities'] = len(label_dict) + 1

with tf.device('/gpu:1'):

    #tf Graph input
    inputs = tf.placeholder(tf.float32, (None, model_config['doc_time_step'], model_config['input_dim']), name='input')
    query = tf.placeholder(tf.float32, (None, model_config['query_time_step'], model_config['input_dim']), name='question')
    labels = tf.placeholder(tf.float32, (None, model_config['n_entities']),name='labels')
    learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
    keep_prob = tf.placeholder(tf.float32, name='dropout_prob')


    doc_var_list = [
                    ['d_attw',[2*model_config['ctx_lstm_size'], model_config['attention_mlp_hidden']]],
                    ['q_attw',[2*model_config['question_lstm_size'], model_config['attention_mlp_hidden']]],
                    ['wms',[model_config['attention_mlp_hidden'],1]],
                    ['w_rg',[2*model_config['ctx_lstm_size'], model_config['n_entities'] ]],
                    ['w_ug',[2*model_config['question_lstm_size'], model_config['n_entities'] ]]
                    ]
               
    doc_var = mut.create_var_xavier('Varibles',doc_var_list)#

    x = tf.unstack(inputs, model_config['doc_time_step'], 1)
    q = tf.unstack(query, model_config['query_time_step'], 1)

    with tf.variable_scope("query"):
        with tf.variable_scope("fw"):
            qlstm_fw_cell = tf.contrib.rnn.LSTMCell(model_config['question_lstm_size'], forget_bias=1.0)
            qlstm_fw_cell = tf.contrib.rnn.DropoutWrapper(qlstm_fw_cell, input_keep_prob=keep_prob)

        with tf.variable_scope("bw"):
            qlstm_bw_cell = tf.contrib.rnn.LSTMCell(model_config['question_lstm_size'], forget_bias=1.0)
            qlstm_bw_cell = tf.contrib.rnn.DropoutWrapper(qlstm_bw_cell, input_keep_prob=keep_prob)
        
        doc_net, fw, bw = rnn.static_bidirectional_rnn(qlstm_fw_cell, qlstm_bw_cell, q ,dtype=tf.float32)
        y_q = tf.concat([fw[-1], bw[-1]],1)#


    def get_attention_weight(wym, y_t, wum, u, wms):
        
        m_t = tf.tanh(tf.matmul(y_t, wym) + tf.matmul(u, wum))
        s_t = tf.transpose(tf.matmul(m_t,wms))
        
        return s_t
       
    with tf.variable_scope("document"):
        with tf.variable_scope("fw"):
            lstm_fw_cell = tf.contrib.rnn.LSTMCell(model_config['ctx_lstm_size'], forget_bias=1.0)
            lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, input_keep_prob=keep_prob)
        with tf.variable_scope("bw"):
            lstm_bw_cell = tf.contrib.rnn.LSTMCell(model_config['ctx_lstm_size'], forget_bias=1.0)
            lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell, input_keep_prob=keep_prob)
            
        docout, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)
        
    ##    Attention layer   #


    with tf.name_scope("attention_layer"):
            weight_stack = []
           
            for tstep in range(len(docout)):
                
                d = docout[tstep]
                s_t = get_attention_weight(doc_var['d_attw'],d, doc_var['q_attw'], y_q,doc_var['wms'])
                weight_stack.append(s_t)
           
            S_T = tf.nn.softmax(tf.stack(weight_stack, axis=1), dim=1)
            content = tf.transpose(tf.stack(docout, axis=0),[2,0,1])
            
         
            r = tf.transpose(tf.reduce_sum(tf.multiply(S_T[0], content), axis=1), [1, 0])
               
            logit = tf.tanh(tf.matmul(r,doc_var['w_rg'])  +  tf.matmul(y_q,doc_var['w_ug'])) 



    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits= logit))
    tloss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits= logit))
    solver = tf.train.RMSPropOptimizer(learning_rate = learning_rate, momentum=0.9, decay=0.95).minimize(loss)


tf.summary.scalar("Cross_Entropy",loss, collections=['train'])
tf.summary.scalar("Test_Cross_Entropy",loss, collections=['test'])
merged_summary_train = tf.summary.merge_all('train')
merged_summary_test = tf.summary.merge_all('test')#

config = tf.ConfigProto()
config.gpu_options.allow_growth=True

test_num = 0
pos = 0

with tf.Session(config = config) as sess:
    
    summary_writer = tf.summary.FileWriter(logfile, sess.graph) 
    saver = tf.train.Saver() 
    
    if continue_training !=0:

        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
        continue_training = 0
        
    else:
        sess.run(tf.global_variables_initializer())   

 
    for epoch in range(epoch_n, Nepoch):
        
        
        shufflelist = []


        for i in range(0, len(os.listdir(test_config['data']))//model_config['batch_size'] ):
            
            print("Start Iteration:{}".format(i))
            index = model_config['batch_size']*i
        
            data, label, question, shufflelist = bg.randombatch_epoch(index, data_config, shufflelist, model_config['batch_size'] , data_config['label_type'], False)
            feeddict={inputs: data, query:question, labels:label, keep_prob:1.0}
            output = sess.run(tf.nn.softmax(logit), feeddict)
            
            for i in range(len(output)):
            
                test_num = test_num + 1
                if np.argmax(label[i]) == np.argmax(output[i]):
                    pos = pos + 1
                else:
                    pos = pos


#    print("Accuracy:{}".format(pos/test_num))
        
    
    summary_writer.close()
    sess.close()  

    
   
    
