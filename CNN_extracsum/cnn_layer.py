import tensorflow as tf
import numpy as np
from scipy.misc import imread
from scipy.misc import imresize

encode_length = 128


im1 = (imread("laska.png")[:,:,:3]).astype(np.float32)
im1 = im1 - np.mean(im1)
im1[:, :, 0], im1[:, :, 2] = im1[:, :, 2], im1[:, :, 0]

im1r = imresize(im1,[encode_length,200])

def conv2d(name, input, w,b, stride, pad='SAME'):
	return tf.nn.relu(tf.nn.bias_add(
		tf.nn.conv2d(input, w, strides=stride, padding=pad, name=name), b))

def maxpool(name, input, stride, pad='VALID'):
	return tf.nn.max_pool(input, ksize=stride, strides=stride,padding=pad, name=name)

weight = {
	"w1": tf.Variable(tf.random_normal([encode_length,128,3,32])),
    "w2": tf.Variable(tf.random_normal([3200,2]))
}

biases = {
    "b1": tf.Variable(tf.random_normal([32])),
    "b2": tf.Variable(tf.random_normal([2]))
}


x = tf.placeholder(tf.float32, [None, 128,200,3])
conv1 = conv2d('conv1',x ,weight["w1"], biases["b1"], [1,128,1,1])
pool1 = maxpool("maxpool1", conv1, [1,1,1,2])
pool1_r = tf.reshape(pool1, [-1,3200])
fn1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(pool1_r, weight["w2"]),biases["b2"]),name="fc1")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    res = sess.run(fn1, feed_dict={x:[im1r]})