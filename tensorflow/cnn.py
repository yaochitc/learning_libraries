import tensorflow as tf

x = tf.placeholder('float', [None, 28, 28])
y = tf.placeholder('float', [None, 10])

tf.nn.conv1d