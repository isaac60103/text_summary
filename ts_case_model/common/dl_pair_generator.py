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

def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
       
    


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

src_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/train'

final_wdict_path ='/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/processed_v2/final_wdict100k.pickle'
final_ldict_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/processed_v2/final_ldict_general.pickle'
w2v_dict_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/w2vec/w2v_dict100k.pickle'
wdict = loadfrompickle(final_wdict_path)
ldict = loadfrompickle(final_ldict_path)
w2v_dict = loadfrompickle(w2v_dict_path)

dir_list = os.listdir(src_path)

TIME_STEP = 500
EMBEDDING_SIZE = len(w2v_dict[0])
LABEL_SIZE = len(ldict)


tf_record_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/toy_test/test'
fidx = len(os.listdir(tf_record_path))
fpath = os.path.join(tf_record_path, 'dl_pair_ts500_'+str(fidx)+ '.tfrecords')
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
    
    if(len(words)!=0):
        #--------------------Encode Words----------------
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
             
        
        #--------------------Encode label---------------- 
        enc_label = encode_labels(ldict, label)
            
     #------------Pack dql pair------------------
        for i in range(len(encode_word)):
        
           
                
            example = tf.train.Example(features=tf.train.Features(feature={
                    
                    'content':_bytes_feature(encode_word[i].tostring()),
                    'label':_bytes_feature(enc_label.tostring())
                    
                    }))     
            writer.write(example.SerializeToString())
    
            
        if idx%32 == 0 and idx != 0:   
            writer.close()   
            
            fidx = len(os.listdir(tf_record_path))
            fpath = os.path.join(tf_record_path, 'dl_pair_ts500_'+str(fidx)+ '.tfrecords')
            writer = tf.python_io.TFRecordWriter(fpath)
    
writer.close()  
    
    



















   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
