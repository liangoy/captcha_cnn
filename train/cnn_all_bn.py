import tensorflow as tf
from util.image import generate_image
from util import ml

batch_size = 128

shape = [batch_size] + list(generate_image(1)[1][0].shape)
print(shape)

x = tf.placeholder(shape=shape, dtype=tf.float32)
y_ = tf.placeholder(shape=[batch_size, 4], dtype=tf.int32)
training = tf.placeholder(dtype=tf.bool)

#X=tf.layers.batch_normalization(x,axis=-1,training=training)

c1 = ml.conv2d(x, conv_filter=[3, 4, 3, 8], ksize=[1, 4, 5, 1], pool_stride=[1, 3, 4, 1],
               bn_training=training)  # [20,40 ]
c2 = ml.conv2d(c1, conv_filter=[3, 4, 8, 16], ksize=[1, 4, 5, 1], pool_stride=[1, 2, 4, 1],
               bn_training=training)  # [10,10]
c3 = ml.conv2d(c2, conv_filter=[3, 4, 16, 32], ksize=[1, 3, 3, 1], pool_stride=[1, 2, 2, 1],
               bn_training=training)  # [5,5]

w = tf.Variable(tf.random_uniform([5, 5, 32, 128], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([128], -1.0, 1.0))
c4 = tf.nn.conv2d(c3, filter=w, strides=[1, 1, 1, 1], padding='VALID')

out =tf.reshape(c4, shape=[batch_size, 128])

y0 = tf.nn.softmax(ml.layer_basic(out, 36))
y1 = tf.nn.softmax(ml.layer_basic(out, 36))
y2 = tf.nn.softmax(ml.layer_basic(out, 36))
y3 = tf.nn.softmax(ml.layer_basic(out, 36))

loss0 = -tf.reduce_mean(tf.one_hot(y_[:, 0], depth=36) * tf.log(y0 + 0.0001)) / batch_size / tf.log(2.0)
loss1 = -tf.reduce_mean(tf.one_hot(y_[:, 1], depth=36) * tf.log(y1 + 0.0001)) / batch_size / tf.log(2.0)
loss2 = -tf.reduce_mean(tf.one_hot(y_[:, 2], depth=36) * tf.log(y2 + 0.0001)) / batch_size / tf.log(2.0)
loss3 = -tf.reduce_mean(tf.one_hot(y_[:, 3], depth=36) * tf.log(y3 + 0.0001)) / batch_size / tf.log(2.0)

loss = loss0 + loss1 + loss2 + loss3

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss0)

y_out0 = tf.argmax(y0, axis=1)
y_out1 = tf.argmax(y1, axis=1)
y_out2 = tf.argmax(y2, axis=1)
y_out3 = tf.argmax(y3, axis=1)

y_out0_ = y_[:, 0]
y_out1_ = y_[:, 1]
y_out2_ = y_[:, 2]
y_out3_ = y_[:, 3]

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(10 ** 10):
    train_y, train_x = generate_image(batch_size)
    sess.run(optimizer, feed_dict={x: train_x, y_: train_y,training:True})
    if i % 10 == 0:
        test_y, test_x = generate_image(batch_size)
        test_loss, y_test0, y_test1, y_test2, y_test3, y_test0_, y_test1_, y_test2_, y_test3_ = sess.run(
            [loss0, y_out0, y_out1, y_out2, y_out3, y_out0_, y_out1_, y_out2_, y_out3_],
            feed_dict={x: test_x, y_: test_y,training:False})
        q = len([i0 for i0, i1, i2, i3, j0, j1, j2, j3 in
                 zip(y_test0, y_test1, y_test2, y_test3, y_test0_, y_test1_, y_test2_, y_test3_) if
                 i0 == j0 ]) / batch_size
        print(test_loss, q * 100)
