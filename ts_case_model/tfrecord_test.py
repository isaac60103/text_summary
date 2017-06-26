import tensorflow as tf
import os



def read_and_decode(filename_queue):
    
    reader = tf.TFRecordReader()#
    _, serialized_example = reader.read(filename_queue)#
    features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
        'content': tf.FixedLenFeature([], tf.int64),
        'label': tf.FixedLenFeature([], tf.string),
        })
    
    content = tf.cast(features['content'], tf.int64)
    label = tf.decode_raw(features['label'], tf.float32)
#    question = tf.decode_raw(features['question'], tf.float32)
    
    label = tf.reshape(label, [1946])
#    question = tf.reshape(question, [10,21523])
    
    
    tcontent, tlabel = tf.train.shuffle_batch([content, label],
                                                      batch_size=32,
                                                     capacity=600,
                                                     num_threads=3,
                                                     min_after_dequeue=0)
    
    
    return tcontent, tlabel
   
    
tf_record_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/taskw2v_dlpairs'
tf_record_list = [os.path.join(tf_record_path, f)  for f in os.listdir(tf_record_path)]
filename_queue = tf.train.string_input_producer(tf_record_list, num_epochs=10)   
tcontent, tlabel = read_and_decode(filename_queue)


init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

with tf.Session() as sess:
    
    
    sess.run(init_op)
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)  
    
    try:
        while not coord.should_stop():
#            c = sess.run(tquestion)
            c,t = sess.run([ tcontent, tlabel])
            print("random_batch")
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

coord.request_stop()
coord.join(threads)
    