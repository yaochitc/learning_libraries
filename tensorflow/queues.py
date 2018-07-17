import tensorflow as tf

def simple_shuffle_batch(source, capacity, batch_size=10):
    queue = tf.RandomShuffleQueue(capacity=capacity,
                                  min_after_dequeue=int(0.9*capacity),
                                  shapes=source.shape, dtypes=source.dtype)

    enqueue = queue.enqueue(source)

    num_thread = 4
    qr = tf.train.QueueRunner(queue, [enqueue] * num_thread)

    tf.train.add_queue_runner(qr)

    return queue.dequeue_many(batch_size)

input = tf.constant(list(range(100)))
input = tf.data.Dataset.from_tensor_slices(input)
input = input.make_one_shot_iterator().get_next()

get_batch = simple_shuffle_batch(input, capacity=20)

with tf.train.MonitoredSession() as sess:
    while not sess.should_stop():
        print(sess.run(get_batch))