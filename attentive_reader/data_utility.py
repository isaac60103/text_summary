import numpy as np
import re
import math
import itertools
from collections import Counter
import collections
import random
import tensorflow as tf

def slipt_doc_by_space(filename):
    
    text = open(filename, "r").read()
    r = re.compile('[^-_a-z0-9A-Z]')
    
    text= r.sub(" ", text)
    
#    for i in range(len(text)):
    
    
    splited = re.split('\s', text)
    
    cleantext = list(filter(None, splited))
    
    return cleantext


def create_vocdicts(filelist):
    
    vocab = []
    
    for f in filelist:
        vocab = vocab + slipt_doc_by_space(f)
    
    
    return vocab

def mapword2dict(words, dictionary):
    
    word2dict = words

    for idx in range(len(words)):
        
        try:
            word2dict[idx] = dictionary[words[idx]]
        except:
            word2dict[idx] = 0
    
    word2dict = list(filter(lambda a: a != 0, word2dict))
    
    return word2dict
    

def generate_label(data, dictionary, windowsize = 2):
    
    labels = {}
    
    for vocab in range(len(dictionary)):
        
        labels[vocab] = []
        match = [i for i,x in enumerate(data) if x==vocab]
      
        for m in match:
            for w in range(1,windowsize):
                
                fw = m-w
                bw = m+w
                 
                if fw > 0 : labels[vocab].append(data[fw]) 
                if bw < len(data): labels[vocab].append(data[bw]) 
                    
    return labels


def randombatch(data, batchsize):
    
    batch = np.ndarray(shape=(batchsize), dtype=np.int32)
    labels = np.ndarray(shape=(batchsize, 1), dtype=np.int32)
    
    for i in range(batchsize):
        
        idx = random.randrange(1, len(data)-1)
        batch[i] = idx
        
        labeldize = len(data[idx])
        if labeldize > 1 : labels[i] = data[idx][random.randrange(0, labeldize - 1)]
        else: labels[i] = data[idx][0]
    return batch, labels

def build_dataset(words, n_words):
  """Process raw inputs into a dataset."""
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(n_words - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  
  data = list(filter(lambda a: a != 0, data))
  return data, count, dictionary, reversed_dictionary



























