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
    
 
  label_dict_ori  = create_label_dict(path)
  label_dict = label_dict_ori[3]
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

  return label_dict_ori, all_words, word_dict



def create_train_pair(dl_pair_path, word_label_pair, dictionary, r_dictionary,label_dict = {}):
    
   
    idx = 0
    for d in word_label_pair:
        
        if d in dictionary:
        
            data = dictionary[d]
            
            for l in word_label_pair[d]:
                idx = idx + 1
                dl_pair = [[data], l]
                statics.savetopickle(os.path.join(dl_pair_path, str(idx) + '.pickle'), dl_pair)
                
    if label_dict != {}:
        for l in label_dict[3]:
            
            idx = idx + 1
    
            labelidx = label_dict[3][l] + len(dictionary)
            
            dictionary[l] = labelidx
            r_dictionary[labelidx] = l
            
            label = np.zeros(len(label_dict[3]))
            label[label_dict[3][l]] = 1
            
            dl_pair = [[labelidx], label]
            statics.savetopickle(os.path.join(dl_pair_path, str(idx) + '.pickle'), dl_pair)
            
    return dictionary, r_dictionary
            

def random_batch(dl_pair_path, index, shufflelist): 
    
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
        
    return  batch_d, batch_l, shufflelist
    

model_file = '/home/dashmoment/workspace/text_summary_data/model/task_w2v.ckpt'
checkpoint_dir ='/home/dashmoment/workspace/text_summary_data/model/'

datapath = '/home/dashmoment/dataset/toy_test' 
words_dict_path = '/home/dashmoment/workspace/text_summary_data/data_label_pair/toy_words_dict_for_taskw2v.pickle'
label_dict_path = '/home/dashmoment/workspace/text_summary_data/data_label_pair/toy_label_dict_for_taskw2v.pickle'
word_label_pair_path = '/home/dashmoment/workspace/text_summary_data/data_label_pair/toy_dl_pair_for_taskw2v.pickle'
dl_pair_path = '/home/dashmoment/workspace/text_summary_data/data_label_pair/task_w2v_dl'

#datapath = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/toy_test'
#words_dict_path = '/home/ubuntu/workspace/text_summary_data/data_label_pair/toy_words_dict_for_taskw2v.pickle'
#label_dict_path = '/home/ubuntu/workspace/text_summary_data/data_label_pair/toy_label_dict_for_taskw2v.pickle'
#word_label_pair_path = '/home/ubuntu/workspace/text_summary_data/data_label_pair/toy_dl_pair_for_taskw2v.pickle'
#dl_pair_path = '/home/ubuntu/workspace/text_summary_data/data_label_pair/task_w2v_dl'

vocabulary_size = 552
batch_size = 64

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
dictionary, r_dictionary = create_train_pair(dl_pair_path, word_label_pair, dictionary, r_dictionary,label_dict)

embedding_size = len(label_dict[3])


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
    
    model_range = [0, len(label_dict[0])]
    os_range = [model_range[1], model_range[1]+len(label_dict[1])]
    cat_range = [os_range[1], os_range[1]+len(label_dict[2])]
    
    m_loss = tf.nn.softmax_cross_entropy_with_logits(labels=train_labels[:,model_range[0]:model_range[1]], logits=output[:,model_range[0]:model_range[1]])
    os_loss = tf.nn.softmax_cross_entropy_with_logits(labels=train_labels[:,os_range[0]:os_range[1]], logits=output[:,os_range[0]:os_range[1]])
    cat_loss = tf.nn.softmax_cross_entropy_with_logits(labels=train_labels[:,cat_range[0]:cat_range[1]], logits=output[:,cat_range[0]:cat_range[1]])
    
    loss = tf.reduce_mean(m_loss + os_loss + cat_loss)
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
    
    
    eval_w2v = tf.concat([tf.nn.softmax(output[:, model_range[0]:model_range[1]]),
                          tf.nn.softmax(output[:, os_range[0]:os_range[1]]),
                            tf.nn.softmax(output[:, cat_range[0]:cat_range[1]])], axis=1)

continue_training = 0
init_epoch = 0
config = tf.ConfigProto()
config.gpu_options.allow_growth=True

with tf.Session(config = config) as sess:
    
    saver = tf.train.Saver()
  
    if continue_training !=0:
    
            resaver = tf.train.Saver()
            resaver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
            continue_training = 0
            
    else:
            sess.run(tf.global_variables_initializer())  
 
    
    for epoch in range(init_epoch,2000):
        
        shufflelist = []
        
        
        for idx in range(len(os.listdir(dl_pair_path))//batch_size):
            
            print("Epoch {}, Process {}/{}".format(epoch, idx, len(os.listdir(dl_pair_path))//batch_size))
            
            index = idx*batch_size
            data, label, shufflelist = random_batch(dl_pair_path, index, shufflelist)
            
            sess.run(optimizer, feed_dict={train_inputs:data, train_labels:label})
            
            
            if idx%200 == 0:
                summary_idx = epoch*len(os.listdir(dl_pair_path)) + index
                saver.save(sess, model_file, global_step=summary_idx)
                sloss = sess.run(loss, feed_dict={train_inputs:data, train_labels:label})
                print("Loss:{}".format(sloss))

    idx = 0
    batch = []
    
    w2v_dict = {}
    
    for i in dictionary:
        
        idx = idx + 1
        
        batch = batch + [dictionary[i]]
        
        if idx %batch_size == 0:
            
             w2v_res = sess.run(eval_w2v, feed_dict={train_inputs:batch})
             
             
             for i in range(len(batch)):
                 w2v_dict[r_dictionary[batch[i]]] = [i, w2v_res[i]]
             
             batch = []
            
            
final_embeddings = []
plot_words = []

try:
  # pylint: disable=g-import-not-at-top
  from sklearn.manifold import TSNE
  
  print("Start TSNE")
  tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
  plot_only = 576
  
  l = statics.loadfrompickle(os.path.join(dl_pair_path,'26318.pickle'))
  final_embeddings.append(l[1])
  d = w2v_dict[r_dictionary[598]][1]
  final_embeddings.append(d)
  final_embeddings = np.stack(final_embeddings)
  
  
#  for i in w2v_dict:
#      
#      final_embeddings.append(w2v_dict[i][1])
#      plot_words.append(w2v_dict[i][0])
#      
#  final_embeddings = np.stack(final_embeddings)
#  
  low_dim_embs = tsne.fit_transform(final_embeddings)
  labels = [r_dictionary[598],r_dictionary[598]]
#  labels = [r_dictionary[i] for i in plot_words]
#  labels = labels[:plot_only]
  w2v.plot_with_labels(low_dim_embs, labels, filename='tsne_2000.png')
except ImportError:
  print('Please install sklearn, matplotlib, and scipy to show embeddings.')   
#    em = sess.run(soft_out, feed_dict={train_inputs:batch_d})
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    








