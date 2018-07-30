import tensorflow as tf
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1))
x = tf.placeholder(tf.float32, shape=(1, 2), name='input')
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
print(sess.run(y, feed_dict={x: [[0.7, 0.9]]}),)