import tensorflow.examples.tutorials.mnist.input_data as id
import tensorflow as tf
mnist = id.read_data_sets('MNIST_date/', one_hot=True)
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.maximum(x, W) + b