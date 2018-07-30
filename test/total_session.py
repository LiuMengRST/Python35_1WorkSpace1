import tensorflow as tf
sess = tf.Session()
#x = tf.Variable(tf.zeros([1, 2]))
#y = tf.Variable(tf.zeros([2, 1]))
y = tf.constant([[1.0,0.0],
                 [0.0, 1.0]])
y_ = tf.constant([[1.0,1.0],
                  [0.0, 1.0]])
#y = tf.constant([[1.0],
#                 [2.0]])
z = tf.reduce_sum(tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))
with tf.Session() as sess:
    #x.initializer.run()
    #y.initializer.run()
    print(sess.run(z))