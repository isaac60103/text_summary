import numpy as np
import pickle
import tensorflow as tf


import os

def savetopickle(filepath, obj):
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()
        
def loadfrompickle(filepath):
    with open(filepath, 'rb') as f:
        
        obj = pickle.load(f)
    f.close()    
    return obj


src_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/processed_v2'
final_wdict_path ='/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/processed_v2/final_wdict20k.pickle'
final_ldict_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/processed_v2/final_ldict.pickle'
tf_record_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/taskw2v_dlpairs'

wdict = loadfrompickle(final_wdict_path)
ldict = loadfrompickle(final_ldict_path)

dir_list = os.listdir(src_path)

EMBEDDING_SIZE = len(wdict)
LABEL_SIZE = len(ldict)



fidx = len(os.listdir(tf_record_path))
fpath = os.path.join(tf_record_path, 'tw2vdl_pair_'+str(fidx)+ '.tfrecords')
writer = tf.python_io.TFRecordWriter(fpath)


for idx in range(len(dir_list)):
    
    
    print("Process:{}/{}".format(idx, len(dir_list)))
    dir_path = os.path.join(src_path, dir_list[idx])
    if not os.path.isdir(dir_path): continue
    
    file_list = os.listdir(dir_path)
    
    contents = []
    words = []
    
    for fid in range(len(file_list)) :
        
        if file_list[fid] != 'label.pickle':
            fpath = os.path.join(dir_path, file_list[fid])
            words = words + loadfrompickle(fpath)
            
        else:
            fpath = os.path.join(dir_path, 'label.pickle')
            label = loadfrompickle(fpath)
    
    #--------------------Encode Words----------------
    encode_word = []
    
    for w in words:
        
        if w in wdict: encode_word.append(wdict[w])
        else:encode_word.append(wdict['UNK'])
        
         
    
    #--------------------Encode label---------------- 
    encode_label = []
    label_type = ["OS", "category", "model"]
    
#    lblank_pad = np.zeros(LABEL_SIZE, np.float32)
    
    for ltype in label_type:
        
        if label[ltype][0] in ldict: 
            encode_label.append(ldict[label[ltype][0]])
#            lblank_pad[ldict[label[ltype][0]]] = 1
#            
        else:
#            lblank_pad[ldict["UNKL"]] = 1
            encode_label.append(ldict["UNKL"])
#            
#    encode_label.append(lblank_pad)
            
     #------------Pack dql pair------------------
     
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
       
    for i in range(len(encode_word)):
        
        for j in range(len(encode_label)):
            
            example = tf.train.Example(features=tf.train.Features(feature={
                    
                    'content':_int64_feature(encode_word[i]),
                    'label':_int64_feature(encode_label[j])
                    
                    }))     
            writer.write(example.SerializeToString())
            
    if idx%10000 == 0 and idx != 0:   
        writer.close()   
        
        fidx = len(os.listdir(tf_record_path))
        fpath = os.path.join(tf_record_path, 'tw2vdl_pair_'+str(fidx)+ '.tfrecords')
        writer = tf.python_io.TFRecordWriter(fpath)
    
writer.close()  
    
    



















   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
