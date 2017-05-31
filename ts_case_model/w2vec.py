import numpy as np
import math
import tensorflow as tf
import os
import pickle
from random import shuffle

#import data_utility.dataparser as parser
import word2vec_utility as w2v
import statics

def create_vocab_dict(path, vocabulary_size, valid_size, getall = False):

  files = os.listdir(path)
 
  
  all_words = []
  
  for f in files:
      subfold = os.path.join(path,f)
      subfiles = os.listdir(subfold)
      
      for sf in subfiles:
          if sf != 'label.pickle':
              
              file = os.path.join(subfold,sf)
              print(file)
              all_words = all_words + statics.loadfrompickle(file)
          #else: labelfile_list = labelfile_list + [os.path.join(subfold,sf)]
          
  reverse_dictionary ,data_label = w2v.build_data_label_pair(all_words, vocabulary_size) 

  dl_pairs = []
  for data in  data_label:
      for label in data_label[data]:
          dl_pairs.append([data, label])
 

  return reverse_dictionary, dl_pairs


vocabulary_size = 10000
batch_size = 64
embedding_size = 1024  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.

valid_size = 3     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64    # Number of negative examples to sample.
num_steps = 1000000
vecfilename = 'w2v.pickle'

datapath = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/processed'
w2vfile = '/home/ubuntu/workspace/text_summary_data/w2v.pickle'


graph = tf.Graph()
with graph.as_default():

  # Input data.
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)#

  # Ops and variables pinned to the CPU because of missing GPU implementation
  with tf.device('/gpu:0'):
    # Look up embeddings for inputs.
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

  loss = tf.reduce_mean(
      tf.nn.nce_loss(weights=nce_weights,
                     biases=nce_biases,
                     labels=train_labels,
                     inputs=embed,
                     num_sampled=num_sampled,
                     num_classes=vocabulary_size))#

  # Construct the SGD optimizer using a learning rate of 1.0.
  optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)#

  # Compute the cosine similarity between minibatch examples and all embeddings.
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
  similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)#

continue_training = 0
init_epoch = 0 
nepoch = 5
checkpoint_dir = '/home/ubuntu/workspace/text_summary_data/model'
model_file = "/home/ubuntu/workspace/text_summary_data/model/w2v.ckpt"

d_rdict, d_label = create_vocab_dict(datapath,vocabulary_size, valid_size)

statics.savetopickle("/home/ubuntu/workspace/text_summary_data/dictionary/reverse_dict.pickle", d_rdict)
statics.savetopickle("/home/ubuntu/workspace/text_summary_data/data_label_pair/w2v_dlpair.pickle", d_label)

with tf.Session(graph=graph) as session:
  
  saver = tf.train.Saver()
  
  if continue_training !=0:

        resaver = tf.train.Saver()
        resaver.restore(session, tf.train.latest_checkpoint(checkpoint_dir))
        continue_training = 0
        
  else:
        session.run(tf.global_variables_initializer())  

  print('Initialized')

  average_loss = 0
  
  for epoch in range(init_epoch, nepoch):
         
     batchlist=[]
     
     for i in range(len(d_label)//batch_size):
         
         index = i*batch_size
         summary_idx = len(d_label)//batch_size*epoch + i
         
         print("Epoch: {}, Loop: {}".format(epoch, summary_idx))
             
         batchlist, batch_data, batch_label = w2v.randombatch(d_label, batch_size, index, batchlist)
         feed_dict = {train_inputs: batch_data, train_labels: batch_label}

         session.run([optimizer, loss], feed_dict=feed_dict)
         
         
         if summary_idx % 10 == 0:
             
             saver.save(session, model_file, global_step=summary_idx)
             feed_dict = {train_inputs: batch_data, train_labels: batch_label}
             loss_val = session.run(loss, feed_dict=feed_dict)
          
             print('Average loss at step ', loss_val)
          

  final_embeddings = normalized_embeddings.eval()


try:
  # pylint: disable=g-import-not-at-top
  from sklearn.manifold import TSNE
  
  print("Start TSNE")
  tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
  plot_only = 500
  low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
  labels = [d_rdict[i] for i in range(plot_only)]
  w2v.plot_with_labels(low_dim_embs, labels)
except ImportError:
  print('Please install sklearn, matplotlib, and scipy to show embeddings.')
#with open(vecfilename, 'wb') as handle:
#    pickle.dump(w2v_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)





