import numpy as np
import math
import tensorflow as tf
import os
from random import shuffle

from scipy import spatial
import word2vec_utility as w2v
import statics
 
words_dict_path = '/home/dashmoment/workspace/text_summary_data/data_label_pair/toy_words_dict_for_taskw2v.pickle'
label_dict_path = '/home/dashmoment/workspace/text_summary_data/data_label_pair/toy_label_dict_for_taskw2v.pickle'
word_label_pair_path = '/home/dashmoment/workspace/text_summary_data/data_label_pair/toy_dl_pair_for_taskw2v.pickle'
dl_pair_path = '/home/dashmoment/workspace/text_summary_data/data_label_pair/task_w2v_dl'
w2v_dict_path = 'w2v_dict_2000_2.pickle'


words = statics.loadfrompickle(words_dict_path)
word_label_pair = statics.loadfrompickle(word_label_pair_path)
label_dict = statics.loadfrompickle(label_dict_path)
w2v_dict = statics.loadfrompickle(w2v_dict_path)

sim_list = {}

for w in w2v_dict:
       
    if w in word_label_pair:
    #if w == 'ioLogik':
        avg_sim = 0
        total_N = 0
        
        embed = w2v_dict[w][1]
              
        for label in word_label_pair[w]:
            
            similarity = 1- spatial.distance.cosine(embed, label)
            avg_sim = avg_sim + similarity
            total_N = total_N + 1
#            print(similarity)
#            print (total_N)
        
        sim_list[w] = [total_N, avg_sim/total_N]