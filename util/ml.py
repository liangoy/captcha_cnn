import tensorflow as tf
from scipy.optimize import leastsq
import numpy as np


def bn(x):
    mean, var = tf.nn.moments(x, axes=[0])
    var += 0.1 ** 7
    hat = (x - mean) / tf.sqrt(var)
    return hat


def bn_with_wb(x):
    w = tf.Variable(tf.random_normal([x.shape[1].value], -1.0, 1.0))
    b = tf.Variable(tf.random_normal([x.shape[1].value], -1.0, 1.0))
    return bn(x) * w + b


def layer_basic(x, size=0, with_b=True):
    if not size:
        size = x.shape[1].value
    w = tf.Variable(tf.random_normal([x.shape[1].value, size], -1.0, 1.0,dtype=tf.float32),dtype=tf.float32)
    if with_b:
        b = tf.Variable(tf.random_normal([size], -1.0, 1.0,dtype=tf.float32),dtype=tf.float32)
        return tf.matmul(x, w) + b
    else:
        return tf.matmul(x, w)


def res(x, size=0, with_bn=True):
    if not with_bn:
        res_bn = lambda lay: lay
    else:
        res_bn = bn
    lay1 = tf.nn.relu(layer_basic(res_bn(x), size=size))
    lay2 = tf.nn.relu(layer_basic(res_bn(lay1)))
    lay3 = tf.nn.relu(layer_basic(res_bn(lay2)))
    lay4 = tf.nn.relu(layer_basic(res_bn(lay3)))
    lay5 = tf.nn.relu(layer_basic(res_bn(lay4)))

    if size:
        X = layer_basic(res_bn(x), size)
    else:
        X = x
    return tf.nn.relu(res_bn(lay5) + res_bn(X))


def conv2d(input, conv_filter, stride=[1, 1, 1, 1], padding='SAME', ksize=None, pool_stride=[1, 1, 1, 1],
           pool_padding='SAME', nn=tf.nn.elu):
    w = tf.Variable(tf.random_uniform(conv_filter, -1.0, 1.0,dtype=tf.float32),dtype=tf.float32)
    b = tf.Variable(tf.random_uniform([conv_filter[-1]], -1.0, 1.0,dtype=tf.float32),dtype=tf.float32)
    tmp=tf.nn.conv2d(input, w, strides=stride, padding=padding) + b
    conv2d_out = nn( tmp )
    return tf.nn.max_pool( conv2d_out, ksize=ksize, strides=pool_stride, padding=pool_padding)


def res_modify(x, y, x_to_modify):
    func = lambda p, x: p[0] * x + p[1]
    error = lambda p, x, y: func(p, x) - y
    par = leastsq(error, (0, 0), args=(x, y))[0]
    x_to_modify = np.array(x_to_modify)
    print(par)
    return x_to_modify * par[0] + par[1]
