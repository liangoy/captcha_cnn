import tensorflow as tf
from util.image import generate_image
from util import ml
import numpy as np

batch_size = 128

shape = [batch_size] + list(generate_image(1)[1][0].shape)
print(shape)

x = tf.placeholder(shape=shape, dtype=tf.float32)
y_ = tf.placeholder(shape=[batch_size, 4], dtype=tf.int32)
training = tf.placeholder(dtype=tf.bool)

x_mean=tf.reshape(tf.tile(tf.reduce_mean(x,axis=[1,2]),[60*160]),shape=[batch_size,60,160])

X =tf.nn.relu( tf.abs(x- x_mean)-30)
XX=X/(X+0.0001)

c1 = ml.conv2d(tf.expand_dims(XX, axis=-1), conv_filter=[5, 5, 1, 32], ksize=[1, 3, 4, 1], pool_stride=[1, 3, 4, 1],
               nn=tf.nn.relu)  # [20,40 ]
c2 = ml.conv2d(c1, conv_filter=[3, 4, 32, 64], ksize=[1, 2, 4, 1], pool_stride=[1, 2, 4, 1], nn=tf.nn.relu)  # [10,10]
c3 = ml.conv2d(c2, conv_filter=[3, 4, 64, 128], ksize=[1, 2, 2, 1], pool_stride=[1, 2, 2, 1], nn=tf.nn.relu)  # [5,5]

w = tf.Variable(tf.random_uniform([5, 5, 128, 256], -1.0, 1.0))
#b = tf.Variable(tf.random_uniform([512], -1.0, 1.0))
c4 = tf.nn.conv2d(c3, filter=w, strides=[1, 1, 1, 1], padding='VALID')

out = tf.nn.relu(tf.layers.batch_normalization(tf.reshape(c4, shape=[batch_size, 256]), training=training))
y = tf.nn.softmax(ml.layer_basic(out, 36 * 4))
Y_ = tf.reshape(tf.one_hot(y_, depth=36), shape=[batch_size, 36 * 4])
loss = -tf.reduce_sum(Y_ * tf.log(y + 0.001)) / batch_size / tf.log(2.0)
# optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

y_out = tf.argmax(tf.reshape(y, shape=[batch_size, 4, 36]), axis=2)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(10 ** 10):
    train_y, train_x = generate_image(batch_size)
    sess.run(optimizer, feed_dict={x: train_x, y_: train_y, training: True})
    if i % 10 == 0:
        test_y, test_x = generate_image(batch_size)
        test_loss, y_test, y_test_ = sess.run((loss, y_out, y_), feed_dict={x: test_x, y_: test_y, training: False})
        q = sum([1 if j==0 else 0 for j in np.sum(np.abs(y_test-y_test_),axis=-1)])/batch_size
        print(test_loss, q * 100)
