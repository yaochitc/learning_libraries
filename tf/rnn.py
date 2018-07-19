import tf as tf

from tf.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../mnist/", one_hot=True)

training_steps = 10000
batch_size = 128
display_step = 200
learning_rate = 0.01

num_input = 28
timesteps = 28
num_hidden = 128
num_classes = 10

weights = tf.Variable(tf.random_normal([num_hidden, num_classes]))
biases = tf.Variable(tf.random_normal([num_classes]))

x = tf.placeholder(tf.float32, (None, timesteps, num_input))
y = tf.placeholder(tf.float32, (None, num_classes))

cell = tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

outputs, state = tf.nn.static_rnn(cell, tf.unstack(x, timesteps, 1), dtype=tf.float32)

logits = tf.matmul(outputs[-1], weights) + biases
prediction = tf.nn.softmax(logits)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(1, training_steps + 1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape((batch_size, timesteps, num_input))
        feed_dict = {x: batch_x, y: batch_y}
        sess.run(train_op, feed_dict=feed_dict)
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict=feed_dict)
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
