from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mn = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.InteractiveSession()
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
sess.run(tf.global_variables_initializer())
y = tf.nn.softmax(tf.matmul(x, W)+b)
corss_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(corss_entropy)
for i in range(1000):
    batch = mn.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0],y_: batch[1]})
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accurancy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(accurancy.eval(feed_dict={x: mn.test.images, y_: mn.test.labels}))