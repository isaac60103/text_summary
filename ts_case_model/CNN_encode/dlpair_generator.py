import numpy as np
import tensorflow as tf
import math
from random import shuffle
import os
import random
import sys

sys.path.append('../')
import common.statics as stat

MAXLENTH = 10000
word_embedding_size = 1024
word_length = 20

def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
       
def encode_allcontents(words):
    
    word_encode = []
    
    if len(words) > MAXLENTH:
        words = words[:MAXLENTH]
    
    for w in words:
        if w in wdict:
            encode = wdict[w]
        else:
            encode = 0
        word_encode.append(w2v_dict[encode])
        
    while len(word_encode)%MAXLENTH != 0:
         word_encode.append(np.zeros(len(w2v_dict[0])))
    
    word_encode = np.vstack(word_encode)
    enc_contents = np.reshape(word_encode, (word_length,-1, word_embedding_size)).astype(np.float32)
    
    return enc_contents


def encode_labels(label_dict, label):
    
    label_type = ["OS", "category", "model"]
    
    LABEL_SIZE = len(label_dict)
   
    lblank_pad = np.zeros(LABEL_SIZE, np.float32)
    
    for ltype in label_type:
              
        
        if label[ltype][0] in ldict: 
            lblank_pad[ldict[label[ltype][0]]] = 1
            
        else:
            lblank_pad[ldict["UNK_"+ltype]] = 1
    
    encode_label = lblank_pad
#    encode_label = np.reshape(encode_label, (-1, LABEL_SIZE))
    
    return encode_label
    


final_wdict_path ='/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/processed_v2/final_wdict20k.pickle'
final_ldict_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/processed_v2/final_ldict_general.pickle'
encode_dict = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/w2vec/w2v_dict100k.pickle'

wdict = stat.loadfrompickle(final_wdict_path)
ldict = stat.loadfrompickle(final_ldict_path)
w2v_dict = stat.loadfrompickle(encode_dict)

src_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/processed_v2'
dir_list = os.listdir(src_path)

tf_record_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/cnn_enc'
fidx = len(os.listdir(tf_record_path))
fpath = os.path.join(tf_record_path, 'dl_pair_ts100k_'+str(fidx)+ '.tfrecords')
writer = tf.python_io.TFRecordWriter(fpath)

all_content = []
all_label = []

count = 0

for idx in range(15456,len(dir_list)):
    
    print("Process: {}/{}".format(idx, len(dir_list)))
    

    dir_path = os.path.join(src_path, dir_list[idx])
    
    if os.path.isdir(dir_path):
        file_list = os.listdir(dir_path)
        
         
        words = []
        
        for fid in range(len(file_list)) :
                
                if file_list[fid] != 'label.pickle':
                    fpath = os.path.join(dir_path, file_list[fid])
                    words = words + stat.loadfrompickle(fpath)
                    
                else:
                    fpath = os.path.join(dir_path, 'label.pickle')
                    label = stat.loadfrompickle(fpath)
                    
        if len(words)!=0:
            
            if count%32 == 0 and idx != 0:   
                writer.close()   
                fidx = idx//32
                fpath = os.path.join(tf_record_path, 'dl_pair_ts100k_'+str(fidx)+ '.tfrecords')
                print(fpath)
                writer = tf.python_io.TFRecordWriter(fpath)
            
            count = count + 1
            enc_contents = encode_allcontents(words)
            enc_label = encode_labels(ldict, label)
                        
            example = tf.train.Example(features=tf.train.Features(feature={
                            'content':_bytes_feature(enc_contents.tostring()),
                            'label':_bytes_feature(enc_label.tostring())
                            
                            }))     
            writer.write(example.SerializeToString())
        
        
    
writer.close()   






        