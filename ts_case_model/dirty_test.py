import numpy as np
import math
import tensorflow as tf
import os
from random import shuffle
import time

import word2vec_utility as w2v
import statics
import json

def create_label_dict(path):
    
  files = os.listdir(path)
  models = []
  OS = []
  category = []
  labels = []
  
  total_file = len(files)
  process_file = 0

  
  for f in files:
      subfold = os.path.join(path,f)
      subfiles = os.listdir(subfold)
      
      process_file = process_file + 1
      print("create_label_dict: {}/{}".format(process_file,total_file))
      
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

def collect_all_words(path):
    
  all_words = []
  
  files = os.listdir(path)
  total_file = len(files)
  process_file = 0

  for idx in range(total_file):
      
      f = files[idx]
      subfold = os.path.join(path,f)
      subfiles = os.listdir(subfold)
      
      process_file = process_file + 1
      print("create_vocab_dict: {}/{}".format(process_file,total_file))
      
      for sf in subfiles:
           if sf != 'label.pickle':         
              file = os.path.join(subfold,sf)
              words = statics.loadfrompickle(file)
              all_words = all_words + words
              
              
  return all_words
            

def add_label_to_dict(src_path, dl_pair_path, dictionary, r_dictionary):
    
    label_path = '/home/ubuntu/workspace/text_summary_data/task_w2v/data_label_pairs/toytmp_label_dict.pickle'
    if os.path.isfile(label_path):
        label_dict_ori = statics.loadfrompickle(label_path)
    else:
        label_dict_ori  = create_label_dict(src_path)
        statics.savetopickle(label_path, label_dict_ori)
    
    label_dict = label_dict_ori[3]
    for l in label_dict:
             
             labelidx = len(dictionary)
             dictionary[l] = labelidx
             r_dictionary[labelidx] = l
                             
                    
    for l in label_dict:
         
             word_onehot = np.zeros(len(dictionary))
             word_onehot[dictionary[l]] = 1
             label = np.zeros(len(label_dict))
             label[label_dict[l]] = 1
             
             index = len(os.listdir(dl_pair_path))
             dl_pair = [word_onehot, label]
             statics.savetopickle(os.path.join(dl_pair_path, str(index+1) + '.pickle'), dl_pair)
             
    return dictionary, r_dictionary
    

def create_train_pair_by_dict(src_path, all_words, dl_pair_path ,dictionary):
  
    
  label_path = '/home/ubuntu/workspace/text_summary_data/task_w2v/data_label_pairs/tmp_label_dict.pickle'
  if os.path.isfile(label_path):
      label_dict_ori = statics.loadfrompickle(label_path)
  else:
      label_dict_ori  = create_label_dict(src_path)
      statics.savetopickle(label_path, label_dict_ori)
  
  
  label_dict = label_dict_ori[3]
  files = os.listdir(src_path)
  total_file = len(files)
  
  unk_onehot = np.zeros(len(dictionary))
  unk_onehot[0] = 1
  
  count = 0
  for w in dictionary:
      
       count = count + 1
       print("create_train_pair_by_dict:{}/{}".format(count, len(dictionary)) )
      
       word_onehot = np.zeros(len(dictionary))
       word_onehot[dictionary[w]] = 1
      
       for idx in range(total_file):
          
          f = files[idx]
          subfold = os.path.join(src_path,f)
          subfiles = os.listdir(subfold)
          
          l = statics.loadfrompickle(os.path.join(subfold,'label.pickle'))
          label = np.zeros(len(label_dict))
          
          
          if l['model'][0].replace(" ","")  in label_dict: label[label_dict[l['model'][0].replace(" ","")]] = 1
          if l['OS'][0].replace(" ","")  in label_dict: label[label_dict[l['OS'][0].replace(" ","")]] = 1
          if l['category'][0].replace(" ","")  in label_dict: label[label_dict[l['category'][0].replace(" ","")]] = 1
          
          for sf in subfiles: 
              
              if sf != 'label.pickle':
              
                  file = os.path.join(subfold,sf)
                  words = statics.loadfrompickle(file)
                  
                  if w in words: 
                      
                      index = len(os.listdir(dl_pair_path))
                      dl_pair = [word_onehot, label]
                      statics.savetopickle(os.path.join(dl_pair_path, str(index+1) + '.pickle'), dl_pair)
                      
                      index = len(os.listdir(dl_pair_path))
                      dl_pair = [unk_onehot, label]
                      statics.savetopickle(os.path.join(dl_pair_path, str(index+1) + '.pickle'), dl_pair)
                      
                      break

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
        
        batch_d.append(dl[0])
        batch_l.append(dl[1])
        
        
    batch_d = np.stack(batch_d, axis = 0)
    batch_l = np.stack(batch_l, axis = 0)
    return  batch_d, batch_l, shufflelist
    

#model_file = '/home/dashmoment/workspace/text_summary_data/model/m_10_v1/task_w2v.ckpt'
#checkpoint_dir ='/home/dashmoment/workspace/text_summary_data/model/m_10_v1'
#
#datapath = '/home/dashmoment/dataset/toy_test' 
#words_dict_path = '/home/dashmoment/workspace/text_summary_data/data_label_pair/toy_words_dict_for_taskw2v.pickle'
#label_dict_path = '/home/dashmoment/workspace/text_summary_data/data_label_pair/toy_label_dict_for_taskw2v.pickle'
#word_label_pair_path = '/home/dashmoment/workspace/text_summary_data/data_label_pair/toy_dl_pair_for_taskw2v.pickle'
#dl_pair_path = '/home/dashmoment/workspace/text_summary_data/data_label_pair/task_w2v_dl'

model_file = '/home/ubuntu/workspace/text_summary_data/task_w2v/model/epoch_2000/task_w2v.ckpt'
checkpoint_dir ='/home/ubuntu/workspace/text_summary_data/task_w2v/model/epoch_2000'

datapath = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/toy_test'
words_dict_path = '/home/ubuntu/workspace/text_summary_data/task_w2v/data_label_pairs/toy_words_dict_for_taskw2v.pickle'
label_dict_path = '/home/ubuntu/workspace/text_summary_data/task_w2v/data_label_pairs/toy_label_dict_for_taskw2v.pickle'
word_label_pair_path = '/home/ubuntu/workspace/text_summary_data/task_w2v/data_label_pairs/toy_dl_pair_for_taskw2v.pickle'
dl_pair_path = '/home/ubuntu/workspace/text_summary_data/task_w2v/data_label_pairs/toy_task_w2v_dl'

vocabulary_size = 552
batch_size = 64


all_words = collect_all_words(datapath)
d,c,dictionary, r_dictionary = w2v.build_dataset(all_words, vocabulary_size)


dictionary, r_dictionary = add_label_to_dict(datapath, dl_pair_path, dictionary, r_dictionary)     
create_train_pair_by_dict(datapath, all_words, dl_pair_path ,dictionary)             
      
      
    

#if os.path.isfile(words_dict_path) and os.path.isfile(word_label_pair_path) and os.path.isfile(label_dict_path):
#    words = statics.loadfrompickle(words_dict_path)
#    word_label_pair = statics.loadfrompickle(word_label_pair_path)
#    label_dict = statics.loadfrompickle(label_dict_path)
#    
#else:
#    print("Create")
#    label_dict , words, word_label_pair = create_vocab_dict(datapath)
#    statics.savetopickle(label_dict_path, label_dict)
#    statics.savetopickle(words_dict_path, words)
#    statics.savetopickle(word_label_pair_path, word_label_pair)
#   

#d,c,dictionary, r_dictionary = w2v.build_dataset(words, vocabulary_size)
#dictionary, r_dictionary, wlp = create_train_pair(dl_pair_path, word_label_pair, dictionary, r_dictionary,label_dict)
#
#embedding_size = len(label_dict[3])
#one_hot_size = len(dictionary)

#with tf.device('/gpu:1'):
#    
#    train_inputs = tf.placeholder(tf.float32, shape=[batch_size, one_hot_size])
#    train_labels = tf.placeholder(tf.float32, shape=[batch_size, embedding_size])
#    
#    
#    weight = tf.Variable(
#        tf.truncated_normal([one_hot_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
#    biases = tf.Variable(tf.zeros([embedding_size]))
#    
#    layer_1 = tf.add(tf.matmul(train_inputs, weight), biases)
#    output = tf.nn.relu(layer_1)
#    
#    model_range = [0, len(label_dict[0])]
#    os_range = [model_range[1], model_range[1]+len(label_dict[1])]
#    cat_range = [os_range[1], os_range[1]+len(label_dict[2])]
#    
#    m_loss = tf.nn.softmax_cross_entropy_with_logits(labels=train_labels[:,model_range[0]:model_range[1]], logits=output[:,model_range[0]:model_range[1]])
#    os_loss = tf.nn.softmax_cross_entropy_with_logits(labels=train_labels[:,os_range[0]:os_range[1]], logits=output[:,os_range[0]:os_range[1]])
#    cat_loss = tf.nn.softmax_cross_entropy_with_logits(labels=train_labels[:,cat_range[0]:cat_range[1]], logits=output[:,cat_range[0]:cat_range[1]])
#    
#    loss = tf.reduce_mean(m_loss + os_loss + cat_loss)
#    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
#    
#    
#    eval_w2v = tf.concat([tf.nn.softmax(output[:, model_range[0]:model_range[1]]),
#                          tf.nn.softmax(output[:, os_range[0]:os_range[1]]),
#                            tf.nn.softmax(output[:, cat_range[0]:cat_range[1]])], axis=1)
#
#continue_training = 0
#init_epoch = 0
#total_epoch = 2000
#config = tf.ConfigProto()
#config.gpu_options.allow_growth=True
#
#with tf.Session(config = config) as sess:
#    
#    saver = tf.train.Saver()
#  
#    if continue_training !=0:
#            
#            resaver = tf.train.Saver()
#            resaver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
#            continue_training = 0
#            
#    else:
#            sess.run(tf.global_variables_initializer())  
# 
#    
#    for epoch in range(init_epoch,total_epoch):
#        
#        shufflelist = []
#        
#        
#        for idx in range(len(os.listdir(dl_pair_path))//batch_size):
#            
#            print("Epoch {}, Process {}/{}".format(epoch, idx, len(os.listdir(dl_pair_path))//batch_size))
#            
#            index = idx*batch_size
#            data, label, shufflelist = random_batch(dl_pair_path, index, shufflelist)
#            
#            sess.run(optimizer, feed_dict={train_inputs:data, train_labels:label})
#            
#            
#            if idx%400 == 0:
#                summary_idx = epoch*len(os.listdir(dl_pair_path)) + index
#                saver.save(sess, model_file, global_step=summary_idx)
#                sloss = sess.run(loss, feed_dict={train_inputs:data, train_labels:label})
#                print("Loss:{}".format(sloss))
#
#    idx = 0
#    batch = []
#    baatch_idx = []
#    
#    w2v_dict = {}
#    
#    for i in dictionary:
#        
#        idx = idx + 1
#        
#        word_onehot = np.zeros(len(dictionary))
#        word_onehot[dictionary[i]] = 1
#        
#        batch.append(word_onehot)
#        baatch_idx.append(dictionary[i])
#        
#        if idx %batch_size == 0:
#            
#             batch = np.stack(batch, axis = 0)
#            
#             w2v_res = sess.run(eval_w2v, feed_dict={train_inputs:batch})
#                         
#             for j in range(len(batch)):
#                 
#                 w2v_dict[r_dictionary[baatch_idx[j]]] = [batch[j], w2v_res[j]]
#             
#             batch = []
#             baatch_idx = []
#             
#    statics.savetopickle("w2v_dict_2000_2.pickle", w2v_dict)
#
#
#def tsne_plot_one_words(w2v_dict, r_dictionary, label_dict, filename='tsne_1_t.png'):          
#            
#    final_embeddings = []
#    word_label = []
#    similarity_list = {}
#    
#    try:
#      # pylint: disable=g-import-not-at-top
#      from sklearn.manifold import TSNE
#      from scipy import spatial
#      
#      print("Start TSNE")
#      tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=10000)
#      
#      for test_label in  label_dict[3]:
#         
#          w2v_emb = w2v_dict[test_label][1]
#          
#          label = np.zeros(len(w2v_dict[test_label][1]))
#          label[label_dict[3][test_label]] = 1
#          
#          final_embeddings = final_embeddings + [w2v_emb, label]
#          word_label = word_label + [test_label,test_label]
#          similarity = 1- spatial.distance.cosine(w2v_emb, label)
#          similarity_list[test_label] = similarity
#      
#      final_embeddings = np.stack(final_embeddings)
#      word_label = np.stack(word_label)        
#      low_dim_embs = tsne.fit_transform(final_embeddings)
#
#      w2v.plot_with_labels(low_dim_embs, word_label, filename)
#      
#      statics.savetopickle("sim_2000_2.pickle", similarity_list)
#      
#      return similarity_list
#    except ImportError:
#      print('Please install sklearn, matplotlib, and scipy to show embeddings.')   
#  
#    
#    
#def tsne_plot_all_words(w2v_dict, r_dictionary, filename='tsne_all_t.png'):          
#            
#    final_embeddings = []
#    plot_words = []
#    
#    try:
#      # pylint: disable=g-import-not-at-top
#      from sklearn.manifold import TSNE
#        
#      print("Start TSNE")
#      tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=10000)
#      plot_only = 576
#        
#      for i in w2v_dict:
#          
#          final_embeddings.append(w2v_dict[i][1])
#          plot_words.append(i)
#          
#      final_embeddings = np.stack(final_embeddings)
#      
#      low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
#      labels = [i for i in plot_words]
#      labels = labels[:plot_only]
#      w2v.plot_with_labels(low_dim_embs, labels, filename)
#      
#      return plot_words
#      
#      
#    except ImportError:
#      print('Please install sklearn, matplotlib, and scipy to show embeddings.')   
#
#        
#    
#sim = tsne_plot_one_words(w2v_dict, r_dictionary,label_dict, 'tsne_1_2000_2.png')  
#pltw = tsne_plot_all_words(w2v_dict, r_dictionary, 'tsne_all_2000_2.png')
    
    
    
    
    
    
    
    
    
    
    
    








