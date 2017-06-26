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
final_ldict_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/processed_v2/final_ldict.pickle'
wdict = stat.loadfrompickle(final_wdict_path)
ldict = stat.loadfrompickle(final_ldict_path)

src_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/processed_v2'
dir_list = os.listdir(src_path)

all_content = {}
all_label = {}

for i in range(2):

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
        encode = np.zeros(len(wdict))
        if w in wdict:
            encode[wdict[w]] = 1
        else:
            encode[0] = 1
            
        word_encode.append(encode)
        
    while len(word_encode)%10000 != 0:
         word_encode.append(encode)
        
    word_encode = np.vstack(word_encode)
    #encode = np.zeros(len(wdict))
    raw_data = np.reshape(word_encode, (1,20,-1, 21523)) 
    
    all_content[i] = raw_data

    LABEL_SIZE = len(ldict)
    encode_label = []
    
    label_type = ["OS", "category", "model"]
    lblank_pad = np.zeros(LABEL_SIZE, np.float32)
        
    for ltype in label_type:
        
       
        
        if label[ltype][0] in ldict: 
            lblank_pad[ldict[label[ltype][0]]] = 1
    #        encode_label.append(lblank_pad)
        else:
            lblank_pad[ldict["UNKL"]] = 1
    #        encode_label.append(lblank_pad)
    
    encode_label = lblank_pad
    encode_label = np.reshape(encode_label, (-1, LABEL_SIZE))
    
    all_label[i] = encode_label
    
    


inputs = tf.placeholder(tf.float32, (None, 20, 500, 21523), name='input')
labels = tf.placeholder(tf.float32, (None, LABEL_SIZE) ,name='labels')
max_step = tf.placeholder(tf.int64)

    
net = slim.repeat(inputs, 2, slim.conv2d, 2048 , [5, 1], scope='conv1')
net = tf.nn.max_pool(net, [1,3, 1,1], [1,2,1,1], padding='VALID')
net = slim.repeat(net, 2, slim.conv2d, 2048 , [5, 1], scope='conv2')
net = tf.nn.max_pool(net, [1,5, 1,1], [1,3,1,1], padding='VALID')
net = slim.repeat(net, 2, slim.conv2d, 2048 , [2, 1], scope='conv3')
net = tf.nn.max_pool(net, [1,2, 1,1], [1,2,1,1], padding='VALID')


net = slim.repeat(net, 2, slim.conv2d, 2048, [1, 5], scope='conv4')
net = tf.nn.max_pool(net, [1,1, 5,1], [1,1,5,1], padding='VALID')

net = slim.repeat(net, 2, slim.conv2d, 2048, [1, 5], scope='conv5')
net = tf.nn.max_pool(net, [1,1, 5,1], [1,1,5,1], padding='VALID')




            
            
                    

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  
    tconv1 = sess.run(net, feed_dict={inputs:raw_data, max_step:3})
    
    

































        
