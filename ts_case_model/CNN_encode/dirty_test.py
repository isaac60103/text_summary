import numpy as np
import tensorflow as tf
import math
from random import shuffle
import os
import random
import sys

sys.path.append('../')
import common.statics as stat


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
    raw_data = np.reshape(word_encode, (1,10,-1, 21523)) 
    
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
    
    


inputs = tf.placeholder(tf.float32, (None, 10, None, 21523), name='input')
labels = tf.placeholder(tf.float32, (None, LABEL_SIZE) ,name='labels')
max_step = tf.placeholder(tf.int64)

with tf.name_scope("conv1"):
    
            s_h = 1; s_w = 3
            rconv1W = tf.Variable(tf.random_normal([1,3,21523,96],stddev=0.01))
            rconv1b = tf.Variable(tf.random_normal([96],mean= 0,stddev= 0.01)) 
            conv1_in = tf.nn.conv2d(inputs, rconv1W, strides=[1,s_h,s_w,1], padding='SAME')
            conv1_add = tf.nn.bias_add(conv1_in, rconv1b)
            conv1 = tf.nn.relu(conv1_add)
            radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
            lrn1 = tf.nn.local_response_normalization(conv1,
                                                          depth_radius=radius,
                                                          alpha=alpha,
                                                          beta=beta,
                                                          bias=bias)
            
            s_h = 1; s_w = 3
            rconv2W = tf.Variable(tf.random_normal([1,3,96,128],stddev=0.01))
            rconv2b = tf.Variable(tf.random_normal([128],mean= 0,stddev= 0.01)) 
            conv2_in = tf.nn.conv2d(conv1, rconv2W, strides=[1,s_h,s_w,1], padding='SAME')
            conv2_add = tf.nn.bias_add(conv2_in, rconv2b)
            conv2 = tf.nn.relu(conv2_add)
            radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
            lrn1 = tf.nn.local_response_normalization(conv2,
                                                          depth_radius=radius,
                                                          alpha=alpha,
                                                          beta=beta,
                                                          bias=bias)
            
            
                    
                
                
                
#maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, 1, 3, 1], strides=[1, 1, max_step, 1], padding='VALID')

shape = tf.shape(conv1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  
    tconv1 = sess.run(conv1, feed_dict={inputs:raw_data, max_step:3})
    tconv2 = sess.run(conv2, feed_dict={inputs:raw_data, max_step:2})
    

































        
