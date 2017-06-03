import numpy as np
import math
import tensorflow as tf
import os
import pickle
from random import shuffle

import word2vec_utility as w2v
import statics

def create_label_dict(path):
    
  files = os.listdir(path)
  models = []
  OS = []
  category = []
  labels = []

  
  for f in files:
      subfold = os.path.join(path,f)
      subfiles = os.listdir(subfold)
      
      for sf in subfiles:
          if sf == 'label.pickle':

              l = statics.loadfrompickle(os.path.join(subfold,sf))
              models.append(l['model'][0].replace(" ",""))
              OS.append(l['OS'][0].replace(" ",""))
              category.append(l['category'][0].replace(" ",""))

  labels = list(set(models)) + list(set(OS)) + list(set(category))
  
  labels_dict = {}
  
  idx = 0
  
  for i in range(len(labels)):
    
      if labels[i] not in labels_dict: 
          
          labels_dict[labels[i]] = idx
          idx = idx + 1
  
  return list(set(models)), list(set(OS)), list(set(category)), labels_dict


def create_vocab_dict(path, vocabulary_size,getall = False):
    
 
  label_dict  = create_label_dict(path)
  files = os.listdir(path)
  word_dict = {}
  all_words = []

   for f in files:
      subfold = os.path.join(path,f)
      subfiles = os.listdir(subfold)
      
      for sf in subfiles:
          
          l = statics.loadfrompickle(os.path.join(subfold,'label.pickle'))
          label = np.zeros(len(label_dict))
          
          print(l['category'][0].replace(" ",""))
          
          if l['model'][0].replace(" ","")  in label_dict: label[label_dict[l['model'][0].replace(" ","")]] = 1
          if l['OS'][0].replace(" ","")  in label_dict: label[label_dict[l['OS'][0].replace(" ","")]] = 1
          if l['category'][0].replace(" ","")  in label_dict: label[label_dict[l['category'][0].replace(" ","")]] = 1
          
          
          if sf != 'label.pickle':
              
              file = os.path.join(subfold,sf)
              print(file)
              words = statics.loadfrompickle(file)
              all_words = all_words + words

              for w in words:
                  
                  if w not in word_dict:
                      word_dict[w] = []
                  
                 
                  word_dict[w].append(label)

  return label_dict, all_words, word_dict



def create_train_pair(dl_pair_path, word_label_pair, dictionary):
    
   
    idx = 0
    for d in word_label_pair:
        
        data = dictionary[d]
        
        for l in word_label_pair[d]:
            idx = idx + 1
            dl_pair = [[data], l]
            statics.savetopickle(os.path.join(dl_pair_path, str(idx) + '.pickle'), dl_pair)
    

datapath = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/toy_test'
#datapath = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/toy_test'
vocabulary_size = 10000
batch_size = 64
 

#word_label = create_vocab_dict(datapath,vocabulary_size)
#statics.savetopickle('dl_pair_for_taskw2v.pickle', word_label)

words_dict_path = '/home/ubuntu/workspace/text_summary_data/data_label_pair/toy_words_dict_for_taskw2v.pickle'
label_dict_path = '/home/ubuntu/workspace/text_summary_data/data_label_pair/toy_label_dict_for_taskw2v.pickle'
word_label_pair_path = '/home/ubuntu/workspace/text_summary_data/data_label_pair/toy_dl_pair_for_taskw2v.pickle'

if os.path.isfile(words_dict_path) and os.path.isfile(word_label_pair_path) and os.path.isfile(label_dict_path):
    words = statics.loadfrompickle(words_dict_path)
    word_label_pair = statics.loadfrompickle(word_label_pair_path)
    label_dict = statics.loadfrompickle(label_dict_path)
    
else:
    
    label_dict , words, word_label_pair = create_vocab_dict(datapath,vocabulary_size)
    statics.savetopickle(label_dict_path, label_dict)
    statics.savetopickle(words_dict_path, words)
    statics.savetopickle(word_label_pair_path, word_label_pair)
   

d,c,dictionary, r_dictionary = w2v.build_dataset(words, vocabulary_size)

dl_pair_path = '/home/ubuntu/workspace/text_summary_data/data_label_pair/task_w2v_dl'
#create_train_pair(dl_pair_path, word_label_pair, dictionary)

shufflelist = []
index = 0
batch_d = []
batch_l = []

if shufflelist == []:
    
    shufflelist = os.listdir(dl_pair_path)
    shuffle(shufflelist)
    print('Shuffle List')
batch_file = shufflelist[index:index+batch_size]

for f in batch_file:
    
    dl = statics.loadfrompickle(os.path.join(dl_pair_path, f))
    batch_d = batch_d + dl[0]
    batch_l.append(dl[1])

batch_l = np.stack(batch_l)

for l in label_dict:
    
    labelidx = label_dict[l] + len(dictionary)
    dictionary[l] = labelidx
    r_dictionary[labelidx] = l

embedding_size = len(label_dict)


with tf.device('/gpu:0'):
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.float32, shape=[batch_size, embedding_size])
    
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, vocabulary_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)
    
    weight = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
    biases = tf.Variable(tf.zeros([embedding_size]))
    
    layer_1 = tf.add(tf.matmul(embed, weight), biases)
    output = tf.nn.relu(layer_1)
    
    m_loss = tf.nn.softmax_cross_entropy_with_logits(label=train_labels[[:,0:12]], logit=output[:,0:12])
#    r_model = tf.nn.softmax(output[:,0:12])
#    r_os = tf.nn.softmax(output[:,12:17])
#    r_c = tf.nn.softmax(output[:,17:])
    
    soft_out = tf.concat([r_model, r_os, r_c], axis = 1)
    
config = tf.ConfigProto()
config.gpu_options.allow_growth=True

with tf.Session(config = config) as sess:
    
    sess.run(tf.global_variables_initializer())  
    
    em = sess.run(soft_out, feed_dict={train_inputs:batch_d})
    em2 = sess.run(output, feed_dict={train_inputs:batch_d})
    
    
    
    
    
    
    
    
    
    
    
    
    
    








