from __future__ import division

import tensorflow as tf

from .base import Network


def build_fourier_net(in_dim, n_freqs, bw, reg_freqs, reg_out,
                      reg_out_bias=0, scale_reg_by_n=False,
                      init_freqs=None, opt_freqs=True,
                      dtype=tf.float32):
    n_hidden = 2 * n_freqs

    net = Network(in_dim, n_hidden, dtype=dtype)
    inputs = net.inputs
    params = net.params

    # Model parameters
    shape_freqs = [in_dim, n_freqs]
    if init_freqs is None:
        init_freqs = tf.random_normal(shape_freqs, stddev=1/bw, dtype=dtype)
    else:
        init_freqs = tf.constant(init_freqs, shape=shape_freqs, dtype=dtype)
    params['freqs'] = tf.Variable(init_freqs, trainable=opt_freqs)
    params['out'] = tf.Variable(tf.random_normal([n_hidden, 1], dtype=dtype))
    params['out_bias'] = tf.Variable(tf.random_normal([1], dtype=dtype))

    # Random fourier feats of inputs: shape (n_X, 2 * n_freqs)
    angles = tf.matmul(inputs['X'], params['freqs'])
    layer_1 = tf.concat([tf.sin(angles), tf.cos(angles)], 1)

    # Pool bags: shape (n_bags, 2 * n_freqs)
    layer_pool = net.bag_pool_layer(layer_1)

    # Output: linear layer
    out_layer = tf.matmul(layer_pool, params['out'])
    net.output = tf.squeeze(out_layer + params['out_bias'])

    # Loss
    net.early_stopper = tf.reduce_mean(tf.square(net.output - inputs['y']))

    if scale_reg_by_n:
        n = tf.cast(tf.squeeze(tf.shape(net.output), [0]), dtype)
        reg_freqs /= n
        reg_out /= n
        reg_out_bias /= n
    net.loss = (
        net.early_stopper
        + reg_freqs * tf.nn.l2_loss(params['freqs'])
        + reg_out * tf.nn.l2_loss(params['out'])
        + reg_out_bias * tf.nn.l2_loss(params['out_bias'])
    )

    return net
