import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_process 
from tensorflow.contrib import learn


#==================Init Parameter================================

tf.flags.DEFINE_string("possample_file","pos.txt","Pos sample files")
tf.flags.DEFINE_string("negsample_file", "neg.txt","Neg sample files")
tf.flags.DEFINE_integer("val_sample_percentage",0.3,"Validate sample percentage of total sample")

tf.flags.DEFINE_string("filter_size","3,4,5","Conv Filter Size")

#******Sess Setting***********
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

#*******Output Dir**************
timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "log", timestamp))

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

#=====================Prepare Data==========================

#******Embedding Vocab********
X_raw, Y_raw = data_process.load_data_and_labels_notag(FLAGS.possample_file, FLAGS.negsample_file)
max_sentence_length = max([len(x.split(" ")) for x in X_raw] )
vocab_processor = learn.preprocessing.VocabularyProcessor(max_sentence_length)
X_after = np.array(list(vocab_processor.fit_transform(X_raw)))

np.random.seed(10)
shuffle_idx = np.random.permutation(np.arange(len(X_raw)))
X_shuf = X_after[shuffle_idx]
Y_shuf = Y_raw[shuffle_idx]


vocab_processor.save(os.path.join(out_dir,"vocab"))

#********Seperate Data for cross validation******
dev_sample_index = -1 * int(FLAGS.val_sample_percentage * float(len(Y_shuf)))
x_train, x_val = X_shuf[:dev_sample_index], X_shuf[dev_sample_index:]
y_train, y_val = Y_shuf[:dev_sample_index], Y_shuf[dev_sample_index:]
print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_val)))

#=======================Training========================

with tf.Graph().as_default():
    
    session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
    
    sess = tf.Session(config=session_conf)
    
    #******Multiview CNN Model********
  
    #***********Optimizer***********
    global_step = tf.Variable(0, name="global_step", trainable = False)
    optimizer = tf.train.AdagradDAOptimizer(1e-3,global_step = global_step)
    #train_op = optimizer.minimize(0,global_step = global_step)

    

    sess.run(tf.global_variables_initializer())

     # Generate batches
     
     batches = data_process.batch_iter(
             list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
     for batch in batches:
         x_batch, y_batch = zip(*batch) #zip(*batch) : *use to unzip the zipped list






