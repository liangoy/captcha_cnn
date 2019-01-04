import tensorflow as tf
from util.image import generate_number_image
from util import ml
import numpy as np

batch_size = 256

shape = [batch_size] + list(generate_number_image(1)[1][0].shape)
print(shape)

x = tf.placeholder(shape=shape, dtype=tf.float32)
y_ = tf.placeholder(shape=[batch_size, 4], dtype=tf.int32)
training = tf.placeholder(dtype=tf.bool)

x00 = tf.transpose(tf.reshape(tf.tile(x[:, 0, 0], [60 * 160]), shape=[60, 160, batch_size]), [2, 0, 1])

X = x - x00

XX = X / (X + 0.0001)

c1 = ml.conv2d(tf.expand_dims(X, axis=-1), conv_filter=[5, 5, 1, 32], ksize=[1, 3, 4, 1], pool_stride=[1, 3, 4, 1],
               nn=tf.nn.relu)  # [20,40 ]
c2 = ml.conv2d(c1, conv_filter=[3, 4, 32, 64], ksize=[1, 2, 4, 1], pool_stride=[1, 2, 4, 1], nn=tf.nn.relu)  # [10,10]
c3 = ml.conv2d(c2, conv_filter=[3, 4, 64, 128], ksize=[1, 2, 2, 1], pool_stride=[1, 2, 2, 1], nn=tf.nn.relu)  # [5,5]

w = tf.Variable(tf.random_uniform([5, 5, 128, 256], -1.0, 1.0, dtype=tf.float32), dtype=tf.float32)
# b = tf.Variable(tf.random_uniform([256], -1.0, 1.0))
c4 = tf.nn.conv2d(c3, filter=w, strides=[1, 1, 1, 1], padding='VALID')

out = tf.reshape(c4,[batch_size,256])

y = ml.layer_basic(out, 10 * 4)

y0, y1, y2, y3 = tf.nn.softmax(y[:, 0:10]), tf.nn.softmax(y[:, 10:20]), tf.nn.softmax(y[:, 20:30]), tf.nn.softmax(
    y[:, 30:40])
Y = tf.concat([y0, y1, y2, y3], axis=1)

Y_ = tf.cast(tf.reshape(tf.one_hot(y_, depth=10), shape=[batch_size, 10 * 4]),dtype=tf.float32)
print(Y.dtype,Y_.dtype)
loss = -tf.reduce_sum(Y_ * tf.log(Y + 0.001)) / batch_size
# optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
y_out = tf.argmax(tf.reshape(Y, shape=[batch_size, 4, 10]), axis=2)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
import time

'''
warm up
'''
train_y, train_x = generate_number_image(batch_size)
train_x=np.array(train_x,dtype=np.float32)
train_y=np.array(train_y,dtype=np.float32)
for _ in range(10):
    sess.run(optimizer, feed_dict={x: train_x, y_: train_y, training: True})


s = time.time()
for i in range(100):
    sess.run(optimizer, feed_dict={x: train_x, y_: train_y, training: True})
e = time.time()
print(e - s)
