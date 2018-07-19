import tf as tf

dataset1 = tf.data.Dataset.range(100)
iterator = dataset1.make_one_shot_iterator()
next_element = iterator.get_next()

sess = tf.Session()
for i in range(100):
    value = sess.run(next_element)
    assert value == i

max_value = tf.placeholder(tf.int64, shape=[])
dataset = tf.data.Dataset.range(max_value)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

sess.run(iterator.initializer, feed_dict={max_value: 10})
for i in range(10):
    value = sess.run(next_element)
    assert i == value

sess.run(iterator.initializer, feed_dict={max_value: 100})
for i in range(100):
    value = sess.run(next_element)
    assert i == value