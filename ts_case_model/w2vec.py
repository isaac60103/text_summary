import numpy as np
import math
import tensorflow as tf
import os
import pickle
from random import shuffle
import time
import collections
import random

#import data_utility.dataparser as parser
import word2vec_utility as w2v
import statics




datapath = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/processed_v2'
w2vfile = '/home/ubuntu/workspace/text_summary_data/w2v.pickle'
word_pool_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/w2vec/words_pool.pickle'
final_wdict_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/processed_v2/final_wdict20k.pickle'

word_pool = statics.loadfrompickle(word_pool_path)
wdict = statics.loadfrompickle(final_wdict_path)

code_words = []
for i in word_pool:
    if i in wdict : code_words.append(wdict[i])
    else:code_words.append(wdict['UNK'])
    
data = code_words

data_index = len(data) - 3

def generate_batch(batch_size, num_skips, skip_window):
    
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  if data_index + span > len(data):
    data_index = 0
  buffer.extend(data[data_index:data_index + span])

  data_index += span
  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [skip_window]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    if data_index == len(data):
      
      for i in range(span): 
          buffer[i] = data[i]
      print(buffer)
      data_index = span
    else:
      buffer.append(data[data_index])
      
      data_index += 1
  # Backtrack a little bit to avoid skipping words in the end of a batch
  data_index = (data_index + len(data) - span) % len(data)
  return batch, labels   


vocabulary_size = len(wdict)
batch_size = 64
embedding_size = 1024  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.

valid_size = 3     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64    # Number of negative examples to sample.
num_steps = len(data)*2
#vecfilename = 'w2v.pickle'



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
  
  tf.summary.scalar('loss',loss)
  merged_summary = tf.summary.merge_all()


continue_training = 1
init_iter = 3029000
N_iter = 3029001

checkpoint_dir = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/w2vec/model'
model_file = "/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/w2vec/model/w2v.ckpt"
log_dir = "/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/w2vec/log"
encode_dict = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/w2vec/w2v_dict.pickle'


config = tf.ConfigProto()
config.gpu_options.allow_growth=True

with tf.Session(graph=graph, config = config) as session:
    
  summary_writer = tf.summary.FileWriter(log_dir, session.graph)

  
  saver = tf.train.Saver()
  
  if continue_training !=0:

        resaver = tf.train.Saver()
        resaver.restore(session, tf.train.latest_checkpoint(checkpoint_dir))
        continue_training = 0
        
  else:
        session.run(tf.global_variables_initializer())  

  print('Initialized')

  average_loss = 0
  
  for iteration in range(init_iter, N_iter):
      
     print("Training Process:{}/{}".format(iteration, N_iter))
     
     batch_inputs, batch_labels  = generate_batch(batch_size, num_skips, skip_window)
     
     feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
     session.run(optimizer, feed_dict=feed_dict)
         
     if iteration % 1000 == 0:
           
             saver.save(session, model_file, global_step=iteration)
             loss_val,loss_sum = session.run([loss, merged_summary], feed_dict=feed_dict)
             
             summary_writer.add_summary(loss_sum, iteration)
          
             print('Average loss at step {}'.format(loss_val))
          

  final_embeddings = normalized_embeddings.eval()
  
  with open(encode_dict, 'wb') as handle:
    pickle.dump(final_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)

final_rwdict_path ='/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/processed_v2/final_rwdict20k.pickle'
d_rdict = statics.loadfrompickle(final_rwdict_path)

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






