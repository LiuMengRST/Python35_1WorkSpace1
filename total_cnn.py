import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST\\", one_hot = True)
sess = tf.InteractiveSession()

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2_2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
x_img = tf.reshape(x, [-1, 28, 28, 1])
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
res_conv1 = tf.nn.relu(conv2d(x_img, W_conv1) + b_conv1)
res_pool1 = max_pool_2_2(res_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
res_conv2 = tf.nn.relu(conv2d(res_pool1, W_conv2) + b_conv2)
res_pool2 = max_pool_2_2(res_conv2)

W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
res_pool2_resh = tf.reshape(res_pool2, [-1, 7*7*64])
res_fc = tf.nn.relu(tf.matmul(res_pool2_resh, W_fc1)+b_fc1)
keep_prob = tf.placeholder(tf.float32)
res_drop = tf.nn.dropout(res_fc, keep_prob=keep_prob)

W_output = weight_variable([1024, 10])
b = bias_variable([10])
y = tf.nn.softmax(tf.matmul(res_drop, W_output)+b)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))
train_step = tf.train.AdagradOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1),tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

tf.global_variables_initializer().run()

for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100==0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1})
        print(i,'----------------------------',train_accuracy*100,'%')
    train_step.run(feed_dict={x:batch[0], y_:batch[1],keep_prob: 0.5})

train_accuracy = accuracy.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1})
print('-------------------------------------------------------', train_accuracy)