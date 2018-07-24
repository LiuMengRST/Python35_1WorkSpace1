import tensorflow as tf
m1 = tf.constant([[3., 3.]])
m2 = tf.constant([[4.], [2.]])
m3 = tf.matmul(m1, m2)
print(m3)
'''
sess = tf.Session()
res = sess.run(m3)
print(res)
sess.close()
'''
with tf.Session() as sess:
    with tf.device("/gpu:1"):
        res = sess.run(m3)
        print(res)