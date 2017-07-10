import tensorflow as tf
import os


def extract_test_data(filename):
    
    example = tf.train.Example()
    testdata =[]
    for record in tf.python_io.tf_record_iterator(filename):
        example.ParseFromString(record)   
        f = example.features.feature
        content = f['content'].bytes_list.value[0]
        content = tf.decode_raw(content, tf.float32)
        content = tf.reshape(content,[500,1024])
        testdata.append(content)
        
    return testdata


def extract_test_label(filename):
    
    example = tf.train.Example()

    for record in tf.python_io.tf_record_iterator(filename):
        example.ParseFromString(record)   
        f = example.features.feature
        label = f['label'].bytes_list.value[0]
        label = tf.decode_raw(label, tf.float32)
       
        
    return label


    
data_tf_record_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/toy_test/test/data'
test_tf_record_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/toy_test/test/label'
test_file = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/toy_test/test/data/5000O000017gOvv.tfrecords'

dtf_record_list = [os.path.join(data_tf_record_path, f)  for f in os.listdir(data_tf_record_path)]
ltf_record_list = [os.path.join(test_tf_record_path, f)  for f in os.listdir(test_tf_record_path)]




init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

with tf.Session() as sess:
    
    
    sess.run(init_op)
    
    for fid in range(len(dtf_record_list)):
        
        print(dtf_record_list[fid])
        
        
        testdata = extract_test_data(dtf_record_list[fid])
        label = extract_test_label(ltf_record_list[fid])
       
        s = sess.run(testdata)
        l = sess.run(label)
      
   
#    coord = tf.train.Coordinator()
#    threads = tf.train.start_queue_runners(sess=sess, coord=coord)  
#    
#    try:
#        while not coord.should_stop():
#            
#            print(len(s))
#            c,t, f1, f2 = sess.run([dfile,lfile, fname, lfname])
#            print(f1)
#            print(f2)
##            print("random_batch")
#    except tf.errors.OutOfRangeError:
#        print('Done training -- epoch limit reached')
#    finally:
#        coord.request_stop()
#
#coord.request_stop()
#coord.join(threads)
    