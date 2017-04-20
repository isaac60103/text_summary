import numpy as np
import math
import tensorflow as tf
import os
import pickle

import data_utility.dataparser as parser
import data_utility.word2vec_utility as w2v



vocabulary_size = 100
batch_size = 64
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.

valid_size = 3     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64    # Number of negative examples to sample.
num_steps = 100
vecfilename = 'w2v.pickle'

datapath = "/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/summary_case_example/mail"
files = os.listdir(datapath)
filelist = []

for f in files:
    filelist = filelist + [os.path.join(datapath,f)]

vocab = parser.create_vocdicts_files(filelist)
reverse_dictionary ,data_label = w2v.build_data_label_pair(vocab, vocabulary_size,valid_size)


graph = tf.Graph()

with graph.as_default():

  # Input data.
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

  # Ops and variables pinned to the CPU because of missing GPU implementation
  with tf.device('/cpu:0'):
    # Look up embeddings for inputs.
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

  loss = tf.reduce_mean(
      tf.nn.nce_loss(weights=nce_weights,
                     biases=nce_biases,
                     labels=train_labels,
                     inputs=embed,
                     num_sampled=num_sampled,
                     num_classes=vocabulary_size))

  # Construct the SGD optimizer using a learning rate of 1.0.
  optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

  # Compute the cosine similarity between minibatch examples and all embeddings.
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, valid_dataset)
  similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)

  # Add variable initializer.
  init = tf.global_variables_initializer()

# Step 5: Begin training.


with tf.Session(graph=graph) as session:
  # We must initialize all variables before we use them.
  init.run()
  print('Initialized')

  average_loss = 0
  for step in range(num_steps):

    batch_data = w2v.randombatch(data_label, batch_size)
    feed_dict = {train_inputs: batch_data[0], train_labels: batch_data[1]}

    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += loss_val

    if step % 2000 == 0:
      if step > 0:
        average_loss /= 2000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print('Average loss at step ', step, ': ', average_loss)
      average_loss = 0

  final_embeddings = normalized_embeddings.eval()

w2v_dict = {}

for i in range(len(reverse_dictionary)):
    w2v_dict[reverse_dictionary[i]] = final_embeddings[i]
    

    
with open(vecfilename, 'wb') as handle:
    pickle.dump(w2v_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)





