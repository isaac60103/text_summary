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
              
              print(subfold)
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
  
  return labels_dict

def create_vocab_dict(path, vocabulary_size,getall = False):
    
 
  label_dict  = create_label_dict(path)
  files = os.listdir(path)
  word_dict = {}
  
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
              
              for w in words:
                  
                  if w not in word_dict:
                      word_dict[w] = []
                  
                 
                  word_dict[w].append(label)

  return word_dict

datapath = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/processed'
#datapath = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/toy_test'
vocabulary_size = 10000

word_label = create_vocab_dict(datapath,vocabulary_size)
statics.savetopickle('dl_pair_for_taskw2v.pickle', word_label)









