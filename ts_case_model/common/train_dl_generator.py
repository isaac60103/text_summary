import numpy as np
import re
import pickle
import os
import shutil
import statics
import sys
import operator

import collections




def pad_label(ldict, label):
    
    blank_label = np.zeros(len(ldict), np.int64)
    
    if label in ldict:
        blank_label[ldict[label]] = 1
       
    else:
        blank_label[0] = 1
    
    return blank_label


def pad_words(wdict, rawdata_path):
    
    
    
    words = statics.loadfrompickle(rawdata_path)
    encode_w = []
  
    for w in words:
        blank_words = np.zeros(len(wdict), np.int64)
        blank_words[wdict[w]] = 1
        
        encode_w.append(blank_words)
       
    
    
    
    return encode_w

final_ldict_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/processed_v2/final_ldict.pickle'
final_wdict_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/processed_v2/final_wdict20k.pickle'
process_data_root = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/processed_v2'
dlpair_root = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/dl_pair_for_training'

wdict = statics.loadfrompickle(final_wdict_path)
ldict = statics.loadfrompickle(final_ldict_path)

src_list = os.listdir(process_data_root)

dir_path = os.path.join(process_data_root, src_list[1])
file_path = os.listdir(dir_path)

FILE_LENGTH = 500
contents = []




raw_w = []
coded_l = {}


for i in file_path:
    
    path = os.path.join(dir_path, i)
    
    if i != 'label.pickle':
        w = pad_words(wdict, path)
        
        
    else:
        label = statics.loadfrompickle(path)
        
                 
        coded_l['model'] = pad_label(ldict, label['model'][0])         
        coded_l['OS'] = pad_label(ldict,label['OS'][0]) 
        coded_l['category'] = pad_label(ldict, label['category'][0])
        coded_l['all'] = coded_l['model'] + coded_l['OS'] + coded_l['category']
        
        
        if  len(coded_l['all'].nonzero()[0]) != 3 \
            or  len(coded_l['model'].nonzero()[0]) != 1 \
                or  len(coded_l['OS'].nonzero()[0]) != 1 \
                    or  len(coded_l['category'].nonzero()[0]) != 1:
                        print("Label Length not fit")
                        break
                        
                        
   
        
        
        

