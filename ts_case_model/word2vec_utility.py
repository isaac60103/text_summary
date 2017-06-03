import numpy as np
import collections
import random
from random import shuffle
import matplotlib.pyplot as plt

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
  
  #data = list(filter(lambda a: a != 0, data))
  return data, count, dictionary, reversed_dictionary


def build_data_label_pair(words, n_words, windowsize = 2):

    data, count, dictionary, reversed_dictionary = build_dataset(words, n_words)
    pairs = generate_label(data, dictionary,windowsize)

    return reversed_dictionary, pairs

def randombatch(data,batchsize, index,batchlist=[]):
   
    batch = np.ndarray(shape=(batchsize), dtype=np.int32)
    labels = np.ndarray(shape=(batchsize, 1), dtype=np.int32)
    
    if batchlist == []:
      print("New Epoch, random shuffle")
      batchlist = [x for x in range(len(data))]
      shuffle(batchlist)
      index = 0

    
    batch_index = batchlist[index:index+batchsize]

    for i in range(len(batch_index)):
          batch[i] = data[batch_index[i]][0]
          labels[i] = data[batch_index[i]][1]

    
    return batchlist, batch, labels


def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
  assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
  plt.figure(figsize=(18, 18))  # in inches
  for i, label in enumerate(labels):

    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.savefig(filename)