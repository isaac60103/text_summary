import numpy as np
import tensorflow as tf
import math
from random import shuffle
import os
import random
import sys

sys.path.append('../')
import common.statics as stat

tf_record_path ='/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/toy_test/test'

def read_and_decode(filename_queue, batch_size):
    
    reader = tf.TFRecordReader()#
    _, serialized_example = reader.read(filename_queue)#
    features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
        'filename':tf.FixedLenFeature([], tf.string),
        'content': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.string)
        })
    
    filename = features['filename']
    content = tf.decode_raw(features['content'], tf.float32)
    label = tf.decode_raw(features['label'], tf.float32)
    
    filename =  tf.reshape(filename, [1])
    content = tf.reshape(content, [20,500,1024])
    label = tf.reshape(label, [1948])
    
    
    tfilename, tcontent, tlabel = tf.train.shuffle_batch([filename, content, label],
                                                      batch_size=batch_size,
                                                     capacity=600,
                                                     num_threads=3,
                                                     min_after_dequeue=0)
    
    
    return tfilename, tcontent, tlabel

tf_record_list = [os.path.join(tf_record_path, f)  for f in os.listdir(tf_record_path)]
train_filename_queue = tf.train.string_input_producer(tf_record_list, num_epochs=175)   


tfilename, tcontent, tlabel = read_and_decode(train_filename_queue, 10)

init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())


with tf.Session() as sess:
    
    sess.run(init_op)
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)  
    
    try:
        while not coord.should_stop():
    
            filename, content, label = sess.run([tfilename, tcontent, tlabel])
            break
            
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
        
coord.request_stop()
coord.join(threads) 