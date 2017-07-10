import tensorflow as tf
from tensorflow.contrib import rnn
import os
import pickle
import model_utility as mut
import numpy as np


def savetopickle(filepath, obj):
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()
        
def loadfrompickle(filepath):
    with open(filepath, 'rb') as f:
        
        obj = pickle.load(f)
    f.close()    
    return obj


checkpoint_dir = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/ts_case_project/model/attentive_reader_v2/relu/model'
checkpoint_filename = os.path.join(checkpoint_dir, 'attr_vanilla_model.ckpt')

src_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/test'
final_wdict_path ='/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/processed_v2/final_wdict20k.pickle'
final_ldict_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/processed_v2/final_ldict.pickle'
w2v_dict_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/w2vec/w2v_dict.pickle'
wdict = loadfrompickle(final_wdict_path)
ldict = loadfrompickle(final_ldict_path)
w2v_dict = loadfrompickle(w2v_dict_path)


model_config = {}
model_config['input_dim'] = 1024
model_config['doc_time_step'] = 500
model_config['query_time_step'] = 10
model_config['ctx_lstm_size'] = 256
model_config['question_lstm_size'] = 256
model_config['attention_mlp_hidden'] = 100
model_config['batch_size'] = 64
model_config['n_entities'] = len(ldict) 


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
            logit = tf.nn.relu(tf.matmul(r,doc_var['w_rg'])  +  tf.matmul(y_q,doc_var['w_ug'])) 
            
            prediction = tf.nn.softmax(tf.reduce_mean(logit, axis = 0))
            

def generate_wrod_label(file_list):
    
    words = []
    
    for fid in range(len(file_list)) :
        
        if file_list[fid] != 'label.pickle':
            fpath = os.path.join(dir_path, file_list[fid])
            words = words + loadfrompickle(fpath)
            
        else:
            fpath = os.path.join(dir_path, 'label.pickle')
            label = loadfrompickle(fpath)
    

    return words, label


def encode_words(words):
    
    TIME_STEP = model_config['doc_time_step']
    encode_word = []
    portions = len(words)//TIME_STEP
    
    for p in range(portions):
        
        init_idx = p*TIME_STEP
        end_idx = init_idx + TIME_STEP 
        
        tmp_words = words[init_idx:end_idx]
        tmp_res = []
        
        for w in tmp_words:
           
            if w in wdict:
                wid = wdict[w]
                
                tmp_res.append(w2v_dict[wid])
            else:
                tmp_res.append(w2v_dict[0])
                
        tmp_res = np.vstack(tmp_res)
        encode_word.append(tmp_res)
    
    init_idx = portions*TIME_STEP
    res_words =  words[init_idx:] 
    tmp_res = []
    
    for w in res_words:
        
        if w in wdict:
            wid = wdict[w]       
            tmp_res.append(w2v_dict[wid])
        else:
            tmp_res.append(w2v_dict[0])
            
    for res_n in range(TIME_STEP - len(res_words)):
           
         tmp_res.append(w2v_dict[0])
    tmp_res = np.vstack(tmp_res)
    encode_word.append(tmp_res)
    encode_word = np.stack(encode_word)
    
    return encode_word


def encode_query():
    
    QTIME_STEP = model_config['query_time_step']
    query_sentence = [["what", "is", "model"],["what", "is", "OS"],["what", "is", "category"]]
    encode_query = []
    for q in query_sentence:
       tmp_q = []
       
       for w in q:
           
           wid = wdict[w]
           tmp_q.append(w2v_dict[wid])
       
       while len(tmp_q) < QTIME_STEP:
           tmp_q.append(w2v_dict[0])
        
       tmp_q = np.vstack(tmp_q)
           
       encode_query.append(tmp_q)
       
    return encode_query
    



init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

config = tf.ConfigProto()
config.gpu_options.allow_growth=True

dir_list = os.listdir(src_path)

label_type = ["model", "OS", "category"]
true_pos = 0
top_5_true_pos = 0
count = 0
encode_query = encode_query()

with tf.Session(config = config) as sess:
    
    saver = tf.train.Saver() 
    saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
    
    
    for idx in range(len(dir_list)):
        
        print("Evaluation Case No. {}/{}".format(idx, len(dir_list)))
        
        dir_path = os.path.join(src_path, dir_list[idx])
        if not os.path.isdir(dir_path): continue
        file_list = os.listdir(dir_path)
        words,raw_label =  generate_wrod_label(file_list)
        data = encode_words(words)  
        
        
        for i in range(len(label_type)):
            
            count = count + 1
            
            question= np.stack([encode_query[i] for j in range(len(data))])
            
            feeddict={inputs: data, query:question,  keep_prob:1}
            predict_res = sess.run(prediction, feed_dict= feeddict)
                         
            label_words = raw_label[label_type[i]][0]
            label_pad = np.zeros(len(ldict), np.float32)
            
            if label_words in ldict: 
                          
                label_pad[ldict[label_words]] = 1
            else:
                label_pad[ldict['UNKL']] = 1
            
            labels = label_pad            
            predict_sort = np.argsort(predict_res)
            top_1_pre = predict_sort[-1]
            top_5_pre = predict_sort[-5:]
            
            if  np.argmax(labels) ==  top_1_pre:
                true_pos = true_pos + 1
            elif  np.argmax(labels) in top_5_pre:
                top_5_true_pos = top_5_true_pos + 1
        
        print("Top1_Error:{}".format(1-true_pos/count))
        print("Top5_Error:{}".format(1-top_5_true_pos/count))
sess.close()  

    
   
    
