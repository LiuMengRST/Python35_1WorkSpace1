#softmaxregsession练习
import tensorflow as tf
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros[784, 10])
b = tf.Variable(tf.zeros[10])
y = tf.nn.softmax(tf.matmul(x, W)+b)
