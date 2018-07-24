import tensorflow as tf
import numpy as np
x = np.float32(np.random.rand(2, 100))
y1 = np.dot([0.100, 0.200], x) + 0.300
b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
y = tf.matmul(W, x)+b
loss = tf.reduce_mean(tf.square(y-y1))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
for step in range(0, 4000):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b))