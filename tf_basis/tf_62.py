import tensorflow as tf
from numpy.random import RandomState
batch_size = 8
W1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
W2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
x = tf.placeholder(tf.float32, shape=(None, 2), name='x_input')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y_input')
a = tf.matmul(x, W1)
y = tf.matmul(a, W2)
cross_entropy = -tf.reduce_mean(y_*tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(0.1).minimize(cross_entropy)
rand = RandomState(1)
dataSet_size = 128
X = rand.rand(dataSet_size, 2)
Y = [[int(x1 + x2 < 1)] for x1, x2 in X]
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run(W1))
    print(sess.run(W2))
    STEPS = 5000
    for i in range(STEPS):
        start = (i * batch_size) % dataSet_size
        #end = min(start + batch_size, dataSet_size)
        end = start + batch_size
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 1000 == 0:
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X,y_: Y})
            print(total_cross_entropy)
    print(sess.run(W1))
    print(sess.run(W2))