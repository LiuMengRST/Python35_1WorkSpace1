import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data\\", one_hot=True)
sess = tf.InteractiveSession()
in_units = 784
out_nuits = 300
W1 = tf.Variable(tf.truncated_normal([in_units, out_nuits], stddev=0.1))
b1 = tf.Variable(tf.zeros([out_nuits]))
W2 = tf.Variable(tf.zeros([out_nuits, 10]))
b2 = tf.Variable(tf.zeros([10]))
x = tf.placeholder(tf.float32, [None, 784])
keep_prob = tf.placeholder(tf.float32)
res1 = tf.nn.relu(tf.matmul(x, W1)+b1)
res1_drop = tf.nn.dropout(res1, keep_prob)
res2 = tf.nn.softmax(tf.matmul(res1_drop, W2)+b2)
res = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce3_sum(res * tf.log(res2),reduction_indices=[1]))
tran_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)
tf.global_variables_initializer().run()
for i in range(3000):
    bx, by = mnist.train.next_batch(100)
    tran_step.run({x:bx, res:by, keep_prob:0.75})

cp = tf.equal(tf.argmax(res, 1), tf.argmax(res2, 1))
ac = tf.reduce_mean(tf.cast(cp, tf.float32))
print(ac.eval({x: mnist.test.images, res: mnist.test.labels, keep_prob: 1.0}))