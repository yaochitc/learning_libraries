import tf as tf
import numpy as np

sess = tf.InteractiveSession()

NUM_EXAMPLES = 2000
training_inputs = np.random.normal(size=(NUM_EXAMPLES, 1))
noise = np.random.normal(size=(NUM_EXAMPLES, 1))
training_outputs = training_inputs * 3 + 2 + noise

W = tf.Variable(5., name='weight')
B = tf.Variable(10., name='bias')

x = tf.placeholder(tf.float32, shape=(None, 1), name='x')
y = tf.placeholder(tf.float32, shape=(None, 1), name='y')

y_pred = W * x + B

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
loss = tf.reduce_mean(tf.square(y_pred - y))
tf.summary.scalar('loss', loss)

train_op = optimizer.minimize(loss)

sess.run(tf.global_variables_initializer())

# Merge all the summaries and write them out to /tmp/mnist_logs (by default)
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('summary/train',
                                      sess.graph)
test_writer = tf.summary.FileWriter('summary//test')

for i in range(500):
    _, summary = sess.run((train_op, merged), feed_dict={x: training_inputs, y: training_outputs})
    train_writer.add_summary(summary, i)

train_writer.close()
test_writer.close()
