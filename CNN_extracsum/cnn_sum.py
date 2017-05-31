import tensorflow as tf
import numpy as np 
import pickle
import data_helpers as dh
from tensorflow.contrib import learn

#test = ['i am good', 'you are good', 'you are not', 'i am too']
vocab_file = "vocab.pickle"

def sentence2vec(sentences, vocabfile):
       
    sentence = [x.split(" ") for x in sentences]
    max_document_length = max([len(x) for x in sentence])
    
    embeded = np.zeros([len(sentences), max_document_length,128])
    
    with open(vocabfile, 'rb') as handle:
        vocabdict = pickle.load(handle)
        
    for i in range(len(sentence)):
        for j in range(max_document_length):
            
            if j < len(sentence[i]):
                try:
                    embeded[i][j] = vocabdict[sentence[i][j]]
                   
                except:
                     embeded[i][j] = vocabdict['UNK']
#                print(embeded[i][j])
                
        
        
    return embeded




posfile = "../../dataset/movie_review/rt-polarity.pos"
negfile = "../../dataset/movie_review/rt-polarity.neg"
vocab_file = "vocab.pickle"


[data, label] = dh.load_data_and_labels(posfile, negfile)

vec = sentence2vec(data,vocab_file)

num_classes = 2

#vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
#x = np.array(list(vocab_processor.fit_transform(data)))
# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(label)))
x_shuffled = vec[shuffle_indices]
y_shuffled = label[shuffle_indices]


# Split train/test set
# TODO: This is very crude, should use cross-validation
dev_sample_index = -1 * int(0.9 * float(len(label)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
#print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

input_x = tf.placeholder(tf.float32, [None,56,128,1], name="input_x")
input_y = tf.placeholder(tf.float32, [None, y_train.shape[1]], name="input_y")


vocab_size = len(vec[0]) #number of Vocab in Samples
embedding_size = 128

#
with tf.name_scope("embedding"):
            W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),name="W")
#            embedded_chars = tf.nn.embedding_lookup(W, input_x)
            embedded_chars_expanded = tf.expand_dims(x_train, -1)

num_filters = 96
filter_shape = [3, 128, 1, num_filters]
W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")

conv = tf.nn.conv2d(
                    input_x,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="SAME",
                    name="conv")

h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

pooled = tf.nn.max_pool(
                h,
                ksize=[1, 3, 128, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")

h_pool_flat = tf.reshape(pooled, [-1, 96])

with tf.name_scope("dropout"):
    h_drop = tf.nn.dropout(h_pool_flat, 0.5)
    
with tf.name_scope("output"):
            W = tf.Variable(
                tf.random_uniform([96, 2], -1.0, 1.0), name= "w2")
            b = tf.Variable(tf.constant(0.1, shape=[2]), name="b2")
#            l2_loss += tf.nn.l2_loss(W)
#            l2_loss += tf.nn.l2_loss(b)
            scores = tf.nn.xw_plus_b(h_drop, W, b, name="scores")
            predictions = tf.argmax(scores, 1, name="predictions")


with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=input_y)
            loss = tf.reduce_mean(losses)
            #loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(predictions, tf.argmax(input_y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

            
with tf.Session(config=tf.ConfigProto(
allow_soft_placement=True, log_device_placement=True)) as sess:
    init = tf.initialize_all_variables()
    sess.run(init)
    embedded = sess.run(embedded_chars_expanded)
    
    em = sess.run(predictions, feed_dict = {input_x:embedded})
#    fscore = sess.run(accuracy, feed_dict={input_x:embedded,input_y:y_train})
    
    
 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    