import tensorflow as tf

a = tf.Variable(0., name='a')
b = 2 * a
g = tf.gradients(a + b, [a, b], stop_gradients=[a, b])

vars = tf.trainable_variables()

with tf.Session() as sess:
    print(sess.run(g))


