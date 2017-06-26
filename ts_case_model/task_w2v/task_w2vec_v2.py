import numpy as np
import tensorflow as tf
import math
from random import shuffle
import os
import random
import sys
import pickle

sys.path.append('../')
import common.statics as stat
import common.word2vec_utility as w2v

def read_and_decode(filename_queue, batchsize):
    
    reader = tf.TFRecordReader()#
    _, serialized_example = reader.read(filename_queue)#
    features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
        'content': tf.FixedLenFeature([], tf.int64),
        'label': tf.FixedLenFeature([], tf.int64),
        })
    
    content = tf.cast(features['content'], tf.int64)
    label = tf.cast(features['label'], tf.int64)
#    label = tf.decode_raw(features['label'], tf.float32)
#    
#    label = tf.reshape(label, [1946])
    
    
    tcontent, tlabel = tf.train.shuffle_batch([content, label],
                                                      batch_size=batchsize,
                                                     capacity=600,
                                                     num_threads=3,
                                                     min_after_dequeue=0)
    
    
    return tcontent, tlabel


src_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/processed_v2'
final_wdict_path ='/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/processed_v2/final_wdict20k.pickle'
final_ldict_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/processed_v2/final_ldict.pickle'
tf_record_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/taskw2v_dlpairs'


wdict = stat.loadfrompickle(final_wdict_path)
ldict = stat.loadfrompickle(final_ldict_path)


vocabulary_size = len(wdict)
embedding_size =  len(ldict)


batch_size = 64

    
with tf.device('/gpu:1'):  

  # Input data.
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    similarity = tf.placeholder(tf.float32)


  # Ops and variables pinned to the CPU because of missing GPU implementation
  
    # Look up embeddings for inputs.
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    
    
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)
    
    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
    
      
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    
    norm_embed = tf.nn.embedding_lookup(normalized_embeddings, train_inputs)
    

    
loss = tf.reduce_mean(
  tf.nn.nce_loss(weights=nce_weights,
                 biases=nce_biases,
                 labels=train_labels,
                 inputs=embed,
                 num_sampled=batch_size,
                 num_classes=embedding_size))#

  
  # Construct the SGD optimizer using a learning rate of 1.0.
optimizer = tf.train.GradientDescentOptimizer(1e-4).minimize(loss)#

tf.summary.scalar('loss',loss)
tf.summary.scalar('similarity',similarity)

merged_summary = tf.summary.merge_all()


continue_training = 1
iteration = 0


save_root = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/ts_case_project/task_w2v'
checkpoint_dir = os.path.join(save_root,'model')
model_file = os.path.join(checkpoint_dir, "task_w2v.ckpt")
log_dir = os.path.join(save_root,'log')
encode_dict = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/ts_case_project/task_w2v/taskw2v_dict.pickle'

tf_record_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/taskw2v_dlpairs'
tf_record_list = [os.path.join(tf_record_path, f)  for f in os.listdir(tf_record_path)]
filename_queue = tf.train.string_input_producer(tf_record_list, num_epochs=100)   
tcontent, tlabel = read_and_decode(filename_queue,batch_size)
tlabel = tf.expand_dims(tlabel, axis=1)


config = tf.ConfigProto()
config.gpu_options.allow_growth=True

with tf.Session(config = config) as sess:
    
  summary_writer = tf.summary.FileWriter(log_dir, sess.graph)  
  saver = tf.train.Saver()

  init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())
  sess.run(init_op)  
  
  if continue_training !=0:

        resaver = tf.train.Saver()
        resaver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
        continue_training = 0
        

  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord) 
    
  try:
        while not coord.should_stop():
            
            break
            iteration = iteration + 1
            print("Training Process:{}".format(iteration))
            data,label = sess.run([tcontent, tlabel])
            feed_dict = {train_inputs: data, train_labels: label}
            sess.run(optimizer, feed_dict = feed_dict)
            
            
            
            print(data[0], label[0])
            
            if iteration % 1000 == 0:
                saver.save(sess, model_file, global_step=iteration)
                loss_val, tnorm_embed = sess.run([loss, norm_embed], feed_dict=feed_dict)
                
                mean_cosin_sim = np.mean(np.abs([tnorm_embed[i][label[i][0]] for i in range(len(label))]))
                
                feed_dict = {train_inputs: data, train_labels: label, similarity:mean_cosin_sim}
                test_sum = sess.run(merged_summary, feed_dict)
                          
                summary_writer.add_summary(test_sum, iteration)
                print('Loss : {}, similarity:{}'.format(loss_val, mean_cosin_sim))
            
            
  except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
  finally:
        coord.request_stop()



  final_embeddings = normalized_embeddings.eval()
      
  with open(encode_dict, 'wb') as handle:
    pickle.dump(final_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)




final_rwdict_path ='/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/processed_v2/final_rwdict20k.pickle'
d_rdict = stat.loadfrompickle(final_rwdict_path)

for l in l

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






