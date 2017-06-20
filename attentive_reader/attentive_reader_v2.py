import tensorflow as tf
from tensorflow.contrib import rnn
import os
import pickle
import model_utility as mut
import time


def savetopickle(filepath, obj):
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()
        
def loadfrompickle(filepath):
    with open(filepath, 'rb') as f:
        
        obj = pickle.load(f)
    f.close()    
    return obj


def read_and_decode(filename_queue, batch_size):
    
    reader = tf.TFRecordReader()#
    _, serialized_example = reader.read(filename_queue)#
    features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
        'content': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.string),
        'question': tf.FixedLenFeature([], tf.string)
        })
    
    content = tf.decode_raw(features['content'], tf.float32)
    label = tf.decode_raw(features['label'], tf.float32)
    question = tf.decode_raw(features['question'], tf.float32)
    
    content = tf.reshape(content, [500,1024])
    label = tf.reshape(label, [1946])
    question = tf.reshape(question, [10,1024])
    
    
    tcontent, tlabel, tquestion = tf.train.shuffle_batch([content, label, question],
                                                      batch_size=batch_size,
                                                     capacity=600,
                                                     num_threads=3,
                                                     min_after_dequeue=0)
    
    
    return tcontent, tlabel, tquestion

checkpoint_dir = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/ts_case_project/model'
checkpoint_filename = os.path.join(checkpoint_dir, 'test/attr_vanilla_model.ckpt')
logfile = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/ts_case_project/log/test'

final_wdict_path ='/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/processed_v2/final_wdict20k.pickle'
final_ldict_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/processed_v2/final_ldict.pickle'
wdict = loadfrompickle(final_wdict_path)
ldict = loadfrompickle(final_ldict_path)

data_config = {}
test_config = {}
model_config = {}
dql_config = {}

model_config['input_dim'] = 1024
model_config['doc_time_step'] = 500
model_config['query_time_step'] = 10
model_config['ctx_lstm_size'] = 256
model_config['question_lstm_size'] = 256
model_config['attention_mlp_hidden'] = 100
model_config['batch_size'] = 16
model_config['n_entities'] = len(ldict) 

continue_training = 0
epoch_n = 0
Nepoch = 100
save_epoch = 300
test_epoch = 500


with tf.device('/gpu:1'):

  
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
            #S_T = tf.nn.softmax(ost[0])
            content = tf.transpose(tf.stack(docout, axis=0),[2,0,1])
                
            r = tf.transpose(tf.reduce_sum(tf.multiply(S_T[0], content), axis=1), [1, 0])          
            logit = tf.tanh(tf.matmul(r,doc_var['w_rg'])  +  tf.matmul(y_q,doc_var['w_ug'])) 


loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits= logit))
tloss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits= logit))
solver = tf.train.RMSPropOptimizer(learning_rate = learning_rate, momentum=0.9, decay=0.95).minimize(loss)


tf.summary.scalar("Cross_Entropy",loss, collections=['train'])
tf.summary.scalar("Test_Cross_Entropy",loss, collections=['test'])
merged_summary_train = tf.summary.merge_all('train')
merged_summary_test = tf.summary.merge_all('test')


tf_record_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/tfrecord_test/'
tf_record_list = [os.path.join(tf_record_path, f)  for f in os.listdir(tf_record_path)]
train_portion = 0.9*len(tf_record_list)
train_list = tf_record_list[:1]
test_list = tf_record_list[1:]


train_filename_queue = tf.train.string_input_producer(train_list, num_epochs=175)   
test_filename_queue = tf.train.string_input_producer(test_list, num_epochs=175)   

tcontent, tlabel, tquestion = read_and_decode(train_filename_queue, model_config['batch_size'])
testcontent, testlabel, testquestion = read_and_decode(test_filename_queue, 1)

init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
iteration = 0

with tf.Session(config = config) as sess:
    
    summary_writer = tf.summary.FileWriter(logfile, sess.graph) 
    saver = tf.train.Saver() 
    
    if continue_training !=0:

        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
        continue_training = 0
        
    else:
        sess.run(init_op)  
        
          
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)  
    
    try:
        while not coord.should_stop():
            
            iteration = iteration + 1
            print("Iteration:{}".format(iteration))
            
            data,label,question = sess.run([tcontent, tlabel, tquestion])
            
            feeddict={inputs: data, query:question, labels:label, learning_rate:5e-5, keep_prob:0.2}
            sess.run(solver, feed_dict=feeddict)
            
            if iteration%5 == 0:
                 train_loss, summary = sess.run([loss, merged_summary_train], feed_dict=feeddict)
                 summary_writer.add_summary(summary, iteration)
                 print("Train loss:{}".format(train_loss))
                 
                 
            if iteration%6 == 0:
                 data,label,question = sess.run([testcontent, testlabel, testquestion])
                 feeddict={inputs: data, query:question, labels:label, learning_rate:5e-5, keep_prob:1}
                 test_loss, tsummary = sess.run([loss, merged_summary_test], feed_dict=feeddict)
                 summary_writer.add_summary(tsummary, iteration)
                 print("Test loss:{}".format(test_loss))
            
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

        
coord.request_stop()
coord.join(threads)   
summary_writer.close()
sess.close()  

    
   
    
