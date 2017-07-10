import numpy as np
import tensorflow as tf
import math
from random import shuffle
import os
import random
import sys
import time

sys.path.append('../')
import common.statics as stat
import tensorflow.contrib.slim as slim




def read_and_decode(filename_queue, batch_size):
    
    reader = tf.TFRecordReader()#
    _, serialized_example = reader.read(filename_queue)#
    features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
        'content': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.string),
        })
    
    content = tf.decode_raw(features['content'], tf.float32)
    label = tf.decode_raw(features['label'], tf.float32)
    
    content = tf.reshape(content, [20,500,1024])
    label = tf.reshape(label, [1948])
    
    min_after_dequeue = 0
    capacity = min_after_dequeue + 3 * batch_size
    
    tcontent, tlabel = tf.train.shuffle_batch([content, label],
                                                     batch_size=batch_size,
                                                     capacity=16,
                                                     num_threads=3,
                                                     min_after_dequeue=min_after_dequeue)
    
    
    return tcontent, tlabel


Batch_SIZE = 8
N_EPOCH = 100
TRAINTEST_RATIO = 0.8

tf_record_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/cnn_enc'
tf_record_list = [os.path.join(tf_record_path, f)  for f in os.listdir(tf_record_path)]
train_portion = int(TRAINTEST_RATIO*len(tf_record_list))
train_list = tf_record_list[:train_portion]
test_list = tf_record_list[train_portion:]


with tf.name_scope('Train_Batch'):
    train_filename_queue = tf.train.string_input_producer(train_list, num_epochs=N_EPOCH)   
    tftrain_batch, tftrain_labels = read_and_decode(train_filename_queue, Batch_SIZE)


    
    learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
    keep_prob = tf.placeholder(tf.float32, name='dropout_prob')
    is_training = tf.placeholder(tf.bool, name='is_training')
    
    
    def build_model(inputs):
    
            
            net = slim.repeat(inputs, 1, slim.conv2d, 1024 , [5, 1], scope='conv1')
            net = tf.nn.max_pool(net, [1,3, 1,1], [1,2,1,1], padding='VALID', name='pool1')
            net = slim.repeat(net, 1, slim.conv2d, 512 , [5, 1], scope='conv2')
            net = tf.nn.max_pool(net, [1,5, 1,1], [1,3,1,1], padding='VALID', name='pool2')
            net = slim.repeat(net, 1, slim.conv2d, 1024 , [2, 1], scope='conv3')
            net = tf.nn.max_pool(net, [1,2, 1,1], [1,2,1,1], padding='VALID', name='pool3')
            
            
            net = slim.repeat(net, 1, slim.conv2d, 1024, [1, 5], scope='conv4')
            net = tf.nn.max_pool(net, [1,1, 5,1], [1,1,3,1], padding='VALID', name='pool4')
            
            net = slim.repeat(net, 1, slim.conv2d, 1024, [1, 5], scope='conv5')
            net = tf.nn.max_pool(net, [1,1, 5,1], [1,1,3,1], padding='VALID', name='pool5')
            
            net = slim.repeat(net, 1, slim.conv2d, 1024, [1, 5], scope='conv6')
            net = tf.nn.max_pool(net, [1,1, 5,1], [1,1,3,1], padding='VALID', name='pool6')
            
            net = slim.repeat(net, 1, slim.conv2d, 1024, [1, 5], scope='conv7')
            net = tf.nn.max_pool(net, [1,1, 5,1], [1,1,3,1], padding='VALID', name='pool7')
            
            net = tf.reshape(net, [-1, int(np.prod(net.get_shape()[1:]))])
            
            
            fc6W = tf.Variable(tf.random_normal([5120,2048]), tf.float32)
            fc6b = tf.Variable(tf.random_normal([2048]), tf.float32)
            fc7W = tf.Variable(tf.random_normal([2048,2048]))
            fc7b = tf.Variable(tf.random_normal([2048]))
            fc8W = tf.Variable(tf.random_normal([2048,1833]))
            fc8b = tf.Variable(tf.random_normal([1833]))
            
            
            net = tf.nn.relu_layer(net, fc6W, fc6b , name='fc6') 
            net = tf.layers.dropout(net, rate=keep_prob, training=is_training, name='dropout1')
            net = tf.nn.relu_layer(net, fc7W, fc7b, name='fc7')
            net = tf.layers.dropout(net, rate=keep_prob, training=is_training, name='dropout2')
               
            logits = tf.nn.xw_plus_b(net, fc8W, fc8b, name='fc8')  
            return logits
    
    
    def calc_loss(logits, labels):
    
       
        model_label =tf.slice(labels, [0,115],[-1,1833])  
        model_logits = logits
    
        with tf.name_scope('Cross_Entropy_Loss'):
        
           
            loss_model = tf.nn.softmax_cross_entropy_with_logits(labels=model_label, logits=model_logits)
            
            loss = tf.reduce_mean(loss_model)
            
        return loss,  [model_logits], [ model_label]
    
with tf.device('/gpu:0'):
    
    with tf.variable_scope('cnn_reader') as scope:
       
        logits_for_train = build_model(tftrain_batch)


    with tf.name_scope('Optimizer'):
    
        loss, _, _ = calc_loss(logits_for_train, tftrain_labels)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        solver = optimizer.minimize(loss)
        
        
with tf.name_scope('train_summary'):
    
    tf.summary.scalar("Cross_Entropy", loss, collections=['train'])
    merged_summary_train = tf.summary.merge_all('train')    


   
    
model_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/ts_case_project/model/cnn_reader_model'
checkpoint_dir =  os.path.join(model_path, 'model')
checkpoint_filename = os.path.join(checkpoint_dir, 'cnn_reader_v1.ckpt')
logfile = os.path.join(model_path, 'log')


iteration = 3200
continue_training = 1
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction = 0.9

init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

with tf.Session(config=config) as sess:
    
    summary_writer = tf.summary.FileWriter(logfile, sess.graph)
    saver = tf.train.Saver()
    sess.run(init_op)  
    
    if continue_training !=0:
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
        continue_training = 0
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)  
    
    try:
        while not coord.should_stop():
            
            iteration = iteration + 1
            print("Iteration:{}".format(iteration))
            
            s = time.clock()
            feed_dict = {learning_rate:1e-4,  keep_prob:0.5, is_training:True}
            sess.run(solver, feed_dict=feed_dict)
            e = time.clock()

            print("Train time", e-s)
            
            if iteration%400 == 0: #Train summary
                
                train_loss, train_sum = sess.run([loss, merged_summary_train], feed_dict=feed_dict)
                print("Train Loss:{}".format(train_loss))
                summary_writer.add_summary(train_sum, iteration)
                saver.save(sess, checkpoint_filename, global_step=iteration)
                

            
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

        
coord.request_stop()
coord.join(threads)   
summary_writer.close()
sess.close()  

    
    

































        
