import tensorflow as tf
a = tf.constant([1.0, 2.0], name='a')
b = tf.constant([2.0, 3.0], name='a')
result = a + b
with tf.Session() as sess:
    print(result.eval())