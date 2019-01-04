import tensorflow as tf
from util.image import generate_image
from util import ml

batch_size = 256

shape = [batch_size] + list(generate_image(1)[1][0].shape)
print(shape)

x = tf.placeholder(shape=shape, dtype=tf.float32)
y_ = tf.placeholder(shape=[batch_size], dtype=tf.int32)
training = tf.placeholder(dtype=tf.bool)

X = tf.layers.batch_normalization(tf.expand_dims(x, axis=-1), training=True, trainable=False, scale=False, center=False,
                                  axis=0)

c1 = ml.conv2d(X, conv_filter=[5, 5, 1, 32], ksize=[1, 3, 4, 1], pool_stride=[1, 3, 4, 1], nn=tf.nn.relu)  # [20,40 ]
c2 = ml.conv2d(c1, conv_filter=[3, 4, 32, 64], ksize=[1, 2, 4, 1], pool_stride=[1, 2, 4, 1], nn=tf.nn.relu)  # [10,10]
c3 = ml.conv2d(c2, conv_filter=[3, 4, 64, 128], ksize=[1, 2, 2, 1], pool_stride=[1, 2, 2, 1], nn=tf.nn.relu)  # [5,5]

w = tf.Variable(tf.random_uniform([5, 5, 128, 256], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([256], -1.0, 1.0))
c4 = tf.nn.conv2d(c3, filter=w, strides=[1, 1, 1, 1], padding='VALID')

out = tf.nn.relu(tf.layers.batch_normalization(tf.reshape(c4, shape=[batch_size, 256]), training=training))
y = tf.nn.softmax(ml.layer_basic(out, 36))
loss = -tf.reduce_sum(tf.one_hot(y_, depth=36) * tf.log(y + 0.001)) / batch_size / tf.log(2.0)
# optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

y_out = tf.argmax(y, axis=1)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(10 ** 10):
    train_y, train_x = generate_image(batch_size)
    train_y = train_y[:, 1]
    sess.run(optimizer, feed_dict={x: train_x, y_: train_y, training: True})
    if i % 10 == 0:
        test_y, test_x = generate_image(batch_size)
        test_y = test_y[:, 1]
        test_loss, y_test, y_test_ = sess.run((loss, y_out, y_), feed_dict={x: test_x, y_: test_y, training: False})
        q = len([i for i, j in zip(y_test, y_test_) if i == j]) / batch_size
        print(test_loss, q * 100)
