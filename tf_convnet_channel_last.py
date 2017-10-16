import numpy as np
import tensorflow as tf
import time

batch_size = 50
num_tries = 100
random_data = False

if random_data:
    input = tf.placeholder(shape=[batch_size, 300, 300, 3], dtype=tf.float32, name='input')
else:
    input = tf.ones(shape=[batch_size, 300, 300, 3], dtype=tf.float32, name='input')

conv1 = tf.layers.conv2d(input, 3, 7, strides = 1, use_bias=False, data_format='channels_last', padding='same')
conv2 = tf.layers.conv2d(conv1, 3, 7, strides = 1, use_bias=False, data_format='channels_last', padding='same')
conv3 = tf.layers.conv2d(conv2, 3, 7, strides = 1, use_bias=False, data_format='channels_last', padding='same')
conv4 = tf.layers.conv2d(conv3, 16, 7, strides = 2, use_bias=False, data_format='channels_last', padding='same')
conv5 = tf.layers.conv2d(conv4, 32, 7, strides = 2, use_bias=False, data_format='channels_last', padding='same')
conv6 = tf.layers.conv2d(conv5, 64, 7, strides = 1, use_bias=False, data_format='channels_last', padding='same')
conv7 = tf.layers.conv2d(conv6, 64, 3, strides = 1, use_bias=False, data_format='channels_last', padding='same')
conv8 = tf.layers.conv2d(conv7, 64, 3, strides = 2, use_bias=False, data_format='channels_last', padding='same')
conv9 = tf.layers.conv2d(conv8, 64, 3, strides = 2, use_bias=False, data_format='channels_last', padding='same')
conv10 = tf.layers.conv2d(conv9, 128, 3, strides = 2, use_bias=False, data_format='channels_last', padding='same')
conv11 = tf.layers.conv2d(conv10, 128, 3, strides = 2, use_bias=False, data_format='channels_last', padding='same')


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Warmup
    if random_data:
        _inp = np.random.randn(batch_size, 300, 300, 3)
        sess.run(conv11, feed_dict={input: _inp})
    else:
        conv11.eval()

    for i in range(num_tries):
        t0 = time.time()
        for j in range(10):
            if random_data:
                _inp = np.random.randn(batch_size, 300, 300, 3)
                sess.run(conv11, feed_dict={input: _inp})
            else:
                conv11.eval()
        t1 = time.time()
        elapsed_time = t1- t0
        print(elapsed_time)
        if i == 0:
            total_time = elapsed_time
        else:
            total_time = 0.9 * total_time + 0.1 * elapsed_time

    print("")
    print("===============================")
    print("===============================")
    print("")
    print("Avg time (miliseconds):", total_time * 1000)
