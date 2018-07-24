import tensorflow as tf
import numpy as np
b = tf.Variable(tf.zeros([100]))
W = tf.Variable(tf.random_uniform([700, 100], -1, 1))
x=tf.placeholder(tf.float32,shape=(100,1))
relu = tf.nn.relu(tf.matmul(W, x)+b)
with tf.Session() as sess:
    sess.run(W.initializer)
    sess.run(b.initializer)  # 变量初始化
    input = np.float32(np.random.randn(100, 1))
    print(sess.run(relu, feed_dict={x: input}))