import numpy as np
import tensorflow as tf
import math
from random import shuffle
import os
import random
import sys

sys.path.append('../')
import common.statics as stat
import tensorflow.contrib.slim as slim


final_wdict_path ='/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/processed_v2/final_wdict20k.pickle'
final_ldict_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/processed_v2/final_ldict_general.pickle'
encode_dict = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/w2vec/w2v_dict100k.pickle'

wdict = stat.loadfrompickle(final_wdict_path)
ldict = stat.loadfrompickle(final_ldict_path)
w2v_dict = stat.loadfrompickle(encode_dict)

src_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/processed_v2'
dir_list = os.listdir(src_path)

all_content = []
all_label = []

for i in range(5):

    dir_path = os.path.join(src_path, dir_list[i])
    file_list = os.listdir(dir_path)
    
     
    words = []
    
    for fid in range(len(file_list)) :
            
            if file_list[fid] != 'label.pickle':
                fpath = os.path.join(dir_path, file_list[fid])
                words = words + stat.loadfrompickle(fpath)
                
            else:
                fpath = os.path.join(dir_path, 'label.pickle')
                label = stat.loadfrompickle(fpath)
                
                
    word_encode = []
    
    if len(words) > 10000:
        words = words[:10000]
    
    for w in words:
        if w in wdict:
            encode = wdict[w]
        else:
            encode = 0
        word_encode.append(w2v_dict[encode])
        
    while len(word_encode)%10000 != 0:
         word_encode.append(np.zeros(len(w2v_dict[0])))
        
    word_encode = np.vstack(word_encode)
    #encode = np.zeros(len(wdict))
    raw_data = np.reshape(word_encode, (1,20,-1, 1024)) 
    
    all_content.append(raw_data)

    LABEL_SIZE = len(ldict)
    encode_label = []
    
    label_type = ["OS", "category", "model"]
    lblank_pad = np.zeros(LABEL_SIZE, np.float32)
        
    for ltype in label_type:
              
        
        if label[ltype][0] in ldict: 
            lblank_pad[ldict[label[ltype][0]]] = 1
            
#            print(label[ltype][0], ldict[label[ltype][0]])
    #        encode_label.append(lblank_pad)
        else:
            lblank_pad[ldict["UNK_"+ltype]] = 1
    #        encode_label.append(lblank_pad)
    
    encode_label = lblank_pad
    encode_label = np.reshape(encode_label, (-1, LABEL_SIZE))
    
    all_label.append(encode_label)
    
    
all_content = np.vstack(all_content)
all_label =  np.vstack(all_label)


inputs = tf.placeholder(tf.float32, (None, 20, 500, 1024), name='input')
labels = tf.placeholder(tf.float32, (None, LABEL_SIZE) ,name='labels')
max_step = tf.placeholder(tf.int64)

    
net = slim.repeat(inputs, 2, slim.conv2d, 2048 , [5, 1], scope='conv1')
net = tf.nn.max_pool(net, [1,3, 1,1], [1,2,1,1], padding='VALID')
net = slim.repeat(net, 2, slim.conv2d, 2048 , [5, 1], scope='conv2')
net = tf.nn.max_pool(net, [1,5, 1,1], [1,3,1,1], padding='VALID')
net = slim.repeat(net, 2, slim.conv2d, 2048 , [2, 1], scope='conv3')
net = tf.nn.max_pool(net, [1,2, 1,1], [1,2,1,1], padding='VALID')


net = slim.repeat(net, 2, slim.conv2d, 2048, [1, 5], scope='conv4')
net = tf.nn.max_pool(net, [1,1, 5,1], [1,1,3,1], padding='VALID')

net = slim.repeat(net, 2, slim.conv2d, 2048, [1, 5], scope='conv5')
net = tf.nn.max_pool(net, [1,1, 5,1], [1,1,3,1], padding='VALID')

net = slim.repeat(net, 2, slim.conv2d, 2048, [1, 5], scope='conv5')
net = tf.nn.max_pool(net, [1,1, 5,1], [1,1,3,1], padding='VALID')

net = slim.repeat(net, 2, slim.conv2d, 2048, [1, 5], scope='conv5')
net = tf.nn.max_pool(net, [1,1, 5,1], [1,1,3,1], padding='VALID')

flatnet = tf.reshape(net, [-1, int(np.prod(net.get_shape()[1:]))])

fc6W = tf.Variable(tf.random_normal([10240,4096]), tf.float32)
fc6b = tf.Variable(tf.random_normal([4096]), tf.float32)
fc6 = tf.nn.relu_layer(flatnet, fc6W, fc6b)

#fc7
#fc(4096, name='fc7')
fc7W = tf.Variable(tf.random_normal([4096,4096]))
fc7b = tf.Variable(tf.random_normal([4096]))
fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)

#fc8
#fc(1000, relu=False, name='fc8')
fc8W = tf.Variable(tf.random_normal([4096,1948]))
fc8b = tf.Variable(tf.random_normal([1948]))
fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)   

os_label = tf.slice(labels, [0,0],[-1,93])
cat_label = tf.slice(labels, [0,93],[-1,22])
model_label =tf.slice(labels, [0,115],[-1,1833])

os_logits = tf.slice(fc8, [0,0],[-1,93])
cat_logits = tf.slice(fc8, [0,93],[-1,22])
model_logits =tf.slice(fc8, [0,115],[-1,1833])


loss_os = tf.nn.softmax_cross_entropy_with_logits(labels=os_label, logits=os_logits)
loss_cat = tf.nn.softmax_cross_entropy_with_logits(labels=cat_label, logits=cat_logits)
loss_model = tf.nn.softmax_cross_entropy_with_logits(labels=model_label, logits=model_logits)

loss = tf.reduce_mean(loss_os + loss_cat + loss_model)

predict_OS = tf.argmax(tf.nn.softmax(os_logits),1)
predict_cat = tf.argmax(tf.nn.softmax(cat_logits),1)
predict_model = tf.argmax(tf.nn.softmax(model_logits),1)

OS_idx = tf.argmax(os_label,1)
cat_idx = tf.argmax(cat_label,1)
model_idx = tf.argmax(model_label,1)

tp_os = tf.equal(predict_OS, OS_idx)
tp_cat = tf.equal(predict_cat, cat_idx)
tp_model = tf.equal(predict_model, model_idx)

test_tp_os = tf.reduce_sum(tf.cast(tp_os, tf.int32))

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
solver = optimizer.minimize(loss)




with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())  
    
    while(True):
    
        _, tloss, eval_os, eval_cat, eval_model = sess.run([solver,loss, test_tp_os, tp_os, tp_model], feed_dict={inputs:all_content, labels:all_label})
        print(tloss, eval_os,eval_cat, eval_model)

    
    

































        
