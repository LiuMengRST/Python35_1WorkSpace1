import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNST_data\\", one_hot = True)
#print(mnist.train.images.shape, mnist.train.labels.shape)
sess = tf.InteractiveSession()
x  = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W)+b)
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
tf.global_variables_initializer().run()
for i in range(1000):
    batch_x, batch_y = mnist.train.next_batch(100)
    train_step.run({x: batch_x, y_:batch_y})
cp = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
ac = tf.reduce_mean(tf.cast(cp, tf.float32))
print(ac.eval({x: mnist.test.images, y_: mnist.test.labels}))