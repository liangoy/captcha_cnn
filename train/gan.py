import tensorflow as tf
from util.image import generate_number_image
import numpy as np

batch_size = 128
learning_rate = 0.01

clip = [-0.1, 0.1]

image_real = tf.placeholder(shape=[batch_size, 60, 160, 1], dtype=tf.float32)
label_fake = tf.placeholder(shape=[batch_size, 4], dtype=tf.int32)


def z_relu(x, z=1):
    x_ = tf.nn.relu(x - tf.nn.relu(x) + z) - z
    x__ = z - tf.nn.relu(z - x - tf.nn.relu(-x))
    return x_ + x__


def discriminator(images, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        w1 = tf.get_variable('w1',shape=[5,5,1,16],dtype=tf.float32,initializer=tf.random_normal_initializer())
        c1 = tf.nn.elu(tf.nn.conv2d(images, w1, strides=[1, 1, 1, 1], padding='SAME'))
        p1 = tf.nn.max_pool(c1, ksize=[1, 3, 4, 1], strides=[1, 3, 4, 1], padding='SAME')
        w2 = tf.get_variable('w2',shape=[3,4,16,32],dtype=tf.float32,initializer=tf.random_normal_initializer())
        c2 = tf.nn.elu(tf.nn.conv2d(p1, w2, strides=[1, 1, 1, 1], padding='SAME'))
        p2 = tf.nn.max_pool(c2, ksize=[1, 2, 4, 1], strides=[1, 2, 4, 1], padding='SAME')
        w3 = tf.get_variable('w3',shape=[3,4,32,64],dtype=tf.float32,initializer=tf.random_normal_initializer())
        c3 = tf.nn.elu(tf.nn.conv2d(p2, w3, strides=[1, 1, 1, 1], padding='SAME'))
        p3 = tf.nn.max_pool(c3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        w4 = tf.get_variable('w4',shape=[5,5,64,128],dtype=tf.float32,initializer=tf.random_normal_initializer())
        c4 = tf.nn.elu(tf.nn.conv2d(p3, w4, strides=[1, 1, 1, 1], padding='VALID'))
        out = tf.reshape(c4, shape=[batch_size, 128])

        w_dense = tf.get_variable('w_dense',shape=[128,1],dtype=tf.float32,initializer=tf.random_normal_initializer())
        b_dense = tf.get_variable('b_dense',shape=[1],dtype=tf.float32,initializer=tf.random_normal_initializer())
        return tf.matmul(out, w_dense) + b_dense


def generator(labels):
    with tf.variable_scope('generator', reuse=False):
        label_one_hot = tf.reshape(tf.one_hot(labels, 10), shape=[batch_size, 4 * 10])
        c1 = tf.nn.elu(tf.reshape(tf.layers.dense(label_one_hot, 5 * 5 * 64), shape=[-1, 5, 5, 64]))
        c2 = tf.nn.elu(tf.layers.conv2d_transpose(c1, filters=32, kernel_size=[3, 4], strides=2, padding='SAME'))
        c3 = tf.nn.elu(tf.layers.conv2d_transpose(c2, filters=16, kernel_size=[3, 4], strides=[2, 4], padding='SAME'))
        c4 = tf.layers.conv2d_transpose(c3, filters=1, kernel_size=5, strides=[3, 4], padding='SAME')
        return c4


image_fake = generator(label_fake)

d_y_real = discriminator(image_real / 255, False)
d_y_fake = discriminator(image_fake, True)
d_loss = tf.reduce_mean(d_y_fake) - tf.reduce_mean(d_y_real)

g_loss = -tf.reduce_mean(d_y_fake)

t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
g_vars = [var for var in t_vars if var.name.startswith('generator')]

d_train_opt = tf.train.RMSPropOptimizer(learning_rate).minimize(d_loss, var_list=d_vars)
g_train_opt = tf.train.RMSPropOptimizer(learning_rate).minimize(g_loss, var_list=g_vars)
d_clip_opt = [var.assign(tf.clip_by_value(var, clip_value_min=clip[0], clip_value_max=clip[1])) for var in d_vars]

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(25):
    _label_fake, _image_real = generate_number_image(batch_size)
    _image_real = np.expand_dims(_image_real, axis=-1)
    sess.run(d_train_opt, feed_dict={label_fake: _label_fake, image_real: _image_real})
    sess.run(d_clip_opt)

for i in range(10 ** 10):

    _label_fake, _image_real = generate_number_image(batch_size)
    _image_real = np.expand_dims(_image_real, axis=-1)
    sess.run(g_train_opt, feed_dict={label_fake: _label_fake, image_real: _image_real})

    if i % 5 == 0:
        _label_fake, _image_real = generate_number_image(batch_size)
        _image_real = np.expand_dims(_image_real, axis=-1)
        sess.run(d_train_opt, feed_dict={label_fake: _label_fake, image_real: _image_real})
        sess.run(d_clip_opt)

        _label_fake, _image_real = generate_number_image(batch_size)
        _image_real = np.expand_dims(_image_real, axis=-1)
        loss = sess.run([g_loss, d_loss], feed_dict={label_fake: _label_fake, image_real: _image_real})
        print(loss)

if __name__ == '__main__':
    from PIL import Image

    n = sess.run(image_fake, feed_dict={label_fake: _label_fake, image_real: _image_real})[0]
    n = np.reshape(n, [60, 160]) * 255
    Image.fromarray(n).show()
