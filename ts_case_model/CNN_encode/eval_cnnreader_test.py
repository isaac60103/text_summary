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
    filename, serialized_example = reader.read(filename_queue)#
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
    
  
    tcontent, tlabel = tf.train.batch([content, label],
                                                     batch_size=batch_size,
                                                     capacity=32,
                                                     num_threads=1
                                                     )
    
    
    return filename, tcontent, tlabel


Batch_SIZE = 1
N_EPOCH = 1
TRAINTEST_RATIO = 0.9

tf_record_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/cnn_enc'
tf_record_list = [os.path.join(tf_record_path, f)  for f in os.listdir(tf_record_path)]
train_portion = int(TRAINTEST_RATIO*len(tf_record_list))
train_list = tf_record_list[:train_portion]
test_list = tf_record_list[train_portion:]

#stat.savetopickle('t1t2_train_list.pickle', train_list)

with tf.name_scope('Test_Batch'):
    test_filename_queue = tf.train.string_input_producer(test_list, shuffle= False,num_epochs=N_EPOCH)   
    filename, tftest_batch, tftest_labels = read_and_decode(test_filename_queue, Batch_SIZE)

    
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
            fc8W = tf.Variable(tf.random_normal([2048,1948]))
            fc8b = tf.Variable(tf.random_normal([1948]))
            
            
            net = tf.nn.relu_layer(net, fc6W, fc6b , name='fc6') 
            net = tf.layers.dropout(net, rate=keep_prob, training=is_training, name='dropout1')
            net = tf.nn.relu_layer(net, fc7W, fc7b, name='fc7')
            net = tf.layers.dropout(net, rate=keep_prob, training=is_training, name='dropout2')
               
            logits = tf.nn.xw_plus_b(net, fc8W, fc8b, name='fc8')  
            return logits
    
    
    def calc_loss(logits, labels):
    
        os_label = tf.slice(labels, [0,0],[-1,93])
        cat_label = tf.slice(labels, [0,93],[-1,22])
        model_label =tf.slice(labels, [0,115],[-1,1833])
        
        os_logits = tf.slice(logits, [0,0],[-1,93])
        cat_logits = tf.slice(logits, [0,93],[-1,22])
        model_logits =tf.slice(logits, [0,115],[-1,1833])
    
        with tf.name_scope('Cross_Entropy_Loss'):
        
            loss_os = tf.nn.softmax_cross_entropy_with_logits(labels=os_label, logits=os_logits)
            loss_cat = tf.nn.softmax_cross_entropy_with_logits(labels=cat_label, logits=cat_logits)
            loss_model = tf.nn.softmax_cross_entropy_with_logits(labels=model_label, logits=model_logits)
            
            loss = tf.reduce_mean(loss_os + loss_cat + loss_model)
            
        return loss,  [os_logits, cat_logits, model_logits], [os_label, cat_label, model_label]
    
     

with tf.device('/gpu:0'):   
        
        with tf.variable_scope('cnn_reader') as scope:
            scope.reuse_variables()
            logits_for_test = build_model(tftest_batch)
    
            with tf.name_scope('Result_eval'):
            
                test_loss, test_logit, test_labe = calc_loss(logits_for_test, tftest_labels)
                
                predict_OS = tf.argmax(tf.nn.softmax(test_logit[0]),1)
                predict_cat = tf.argmax(tf.nn.softmax(test_logit[1]),1)
                predict_model = tf.argmax(tf.nn.softmax(test_logit[2]),1)
                
                OS_idx = tf.argmax(test_labe[0],1)
                cat_idx = tf.argmax(test_labe[1],1)
                model_idx = tf.argmax(test_labe[2],1)
                
                tp_os = tf.reduce_sum(tf.cast(tf.equal(predict_OS, OS_idx), tf.int32))
                tp_cat = tf.reduce_sum(tf.cast(tf.equal(predict_cat, cat_idx), tf.int32))
                tp_model = tf.reduce_sum(tf.cast(tf.equal(predict_model, model_idx), tf.int32))
    
    
   
   
    
model_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/ts_case_project/model/cnn_reader/'
checkpoint_dir =  os.path.join(model_path, 'model')
checkpoint_filename = os.path.join(checkpoint_dir, 'cnn_reader_v1.ckpt')
logfile = os.path.join(model_path, 'log')


iteration = 0
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction = 0.9

init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

Total_tp_os = 0
Total_tp_cat = 0
Total_tp_model = 0

with tf.Session(config=config) as sess:
    
#    summary_writer = tf.summary.FileWriter(logfile, sess.graph)
    saver = tf.train.Saver()
    sess.run(init_op)    
    saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
        
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)  
    
    try:
        while not coord.should_stop():
            
            iteration = iteration + 1
            print("Iteration:{}".format(iteration))
            feed_dict = { keep_prob:1, is_training:False}
#            sess.run(tftest_batch)
            true_os, true_cat, true_model = sess.run([tp_os, tp_cat,tp_model], feed_dict=feed_dict)
            Total_tp_os = Total_tp_os + true_os
            Total_tp_cat = Total_tp_cat + true_cat
            Total_tp_model = Total_tp_model + true_model
            
            print("Total OS",Total_tp_os/iteration)
            print("Total Category",Total_tp_cat/iteration)
            print("Total Model", Total_tp_model/iteration)
#            if iteration%200 == 0: #Train summary
#                
#                train_loss, train_sum = sess.run([loss, merged_summary_train], feed_dict=feed_dict)
#                print("Train Loss:{}".format(train_loss))
#                summary_writer.add_summary(train_sum, iteration)
#                saver.save(sess, checkpoint_filename, global_step=iteration)
                
#            if iteration%500 == 0: #Test summary
#            
#                          
#                feed_dict = {learning_rate:1e-4,  keep_prob:1, is_training:False}
#                
#                tloss, test_sum = sess.run([test_loss, merged_summary_test], feed_dict=feed_dict)
#                print("Test Loss:{}".format(tloss))
#                summary_writer.add_summary(test_sum, iteration)
                
                
            
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

        
coord.request_stop()
coord.join(threads)   
sess.close()  

    
    

































        
