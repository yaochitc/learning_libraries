import tf as tf

from tf.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../mnist/", one_hot=True)

training_steps = 10000
batch_size = 128
display_step = 200
learning_rate = 0.01

num_hidden = 128
num_classes = 10

images = tf.placeholder('float', [None, 28, 28, 1])
labels = tf.placeholder('float', [None, num_classes])

with tf.variable_scope('conv1'):
    kernel = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1, dtype=tf.float32), name='weight')
    bias = tf.Variable(tf.zeros([32]), dtype=tf.float32, name='bias')
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], 'SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, bias))
    pool = tf.nn.max_pool(relu, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

with tf.variable_scope('conv2'):
    kernel = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1, dtype=tf.float32), name='weight')
    bias = tf.Variable(tf.zeros([64]), dtype=tf.float32, name='bias')
    conv2 = tf.nn.conv2d(pool, kernel, [1, 1, 1, 1], 'SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv2, bias))
    pool2 = tf.nn.max_pool(relu, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

with tf.variable_scope('local3') as scope:
    reshape = tf.contrib.layers.flatten(pool2)
    dim = reshape.get_shape()[1].value
    weight = tf.Variable(tf.truncated_normal([dim, 512], dtype=tf.float32), name='weight')
    bias = tf.Variable(tf.zeros([512]), dtype=tf.float32, name='bias')
    local3 = tf.nn.relu(tf.matmul(reshape, weight) + bias, name=scope.name)

with tf.variable_scope('softmax_linear'):
    weight = tf.Variable(tf.truncated_normal([512, num_classes], stddev=0.1, dtype=tf.float32), name='weight')
    bias = tf.Variable(tf.constant(0.1, shape=[num_classes], dtype=tf.float32))
    logits = tf.matmul(local3, weight) + bias

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    labels=labels, logits=logits))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

prediction = tf.nn.softmax(logits)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(1, training_steps + 1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape((batch_size, 28, 28, 1))
        feed_dict = {images: batch_x, labels: batch_y}
        sess.run(train_op, feed_dict=feed_dict)
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict=feed_dict)
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
