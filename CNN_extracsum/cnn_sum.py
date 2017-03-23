import tensorflow as tf
import numpy as np 
import pickle
import data_helpers as dh
from tensorflow.contrib import learn

test = ['i am good', 'you are good']
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

max_document_length = max([len(x.split(" ")) for x in data]) #Find longest Sentence
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(data)))
# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(label)))
x_shuffled = x[shuffle_indices]
y_shuffled = label[shuffle_indices]


# Split train/test set
# TODO: This is very crude, should use cross-validation
dev_sample_index = -1 * int(0.2 * float(len(label)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

input_x = tf.placeholder(tf.int32, [None,x_train.shape[1]], name="input_x")
input_y = tf.placeholder(tf.float32, [None, y_train.shape[1]], name="input_y")


vocab_size = len(vocab_processor.vocabulary_) #number of Vocab in Samples
embedding_size = 128

sentence2vec(x_train, vocab_file)
#
with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),name="W")
            embedded_chars = tf.nn.embedding_lookup(W, input_x)
            embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
            
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    em = sess.run(embedded_chars, feed_dict = {input_x:x_train})