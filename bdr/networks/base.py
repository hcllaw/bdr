from __future__ import division
from collections import namedtuple

import numpy as np
import tensorflow as tf


SparseInfo = namedtuple('SparseInfo', ['indices', 'values', 'dense_shape'])


def sparse_matrix_placeholders(dtype=np.float32):
    '''
    Placeholders for a tf.SparseMatrix; use tf.SparseMatrix(*placeholders)
    to make the actual object.
    '''
    return SparseInfo(
        indices=tf.placeholder(tf.int64, [None, 2]),
        values=tf.placeholder(dtype, [None]),
        dense_shape=tf.placeholder(tf.int64, [2]),
    )


def mean_matrix(feats, sparse=False, dtype=np.float32):
    '''
    Returns a len(feats) x feats.total_pts matrix to do mean-pooling by bag.
    '''
    if sparse:
        bounds = np.r_[0, np.cumsum(feats.n_pts)]
        return SparseInfo(
            indices=np.vstack([
                [i, j]
                for i, (start, end) in enumerate(zip(bounds[:-1], bounds[1:]))
                for j in range(start, end)
            ]),
            values=[1/len(bag) for bag in feats for _ in range(len(bag))],
            dense_shape=[len(feats), feats.total_points],
        )
    else:
        mean_mat = np.zeros((len(feats), feats.total_points), dtype=dtype)
        index = 0
        for j in range(len(feats)):
            index_up = index
            index_down = index + feats.n_pts[j]
            mean_mat[j, index_up:index_down] = 1 / feats.n_pts[j]
            index = index_down
        return mean_mat


class Network(object):
    def __init__(self, in_dim, n_hidden, dtype=tf.float32):
        self.in_dim = in_dim
        self.n_hidden = n_hidden
        self.inputs = {
            'X': tf.placeholder(dtype, [None, in_dim]),  # all bags stacked up
            'sizes': tf.placeholder(dtype, [None]),  # one per bag
            'mean_matrix': sparse_matrix_placeholders(dtype),  # n_bags, n_pts
            'in_training': tf.placeholder(tf.bool, shape=[]),  # for batch norm
            'y': tf.placeholder(dtype, [None]),  # one per bag
        }
        self.params = {}
        self.dtype = dtype

    def feed_dict(self, batch, labels=None, training=False):
        batch.make_stacked()
        i = self.inputs
        d = {
            i['X']: batch.stacked_features,
            i['sizes']: batch.n if hasattr(batch, 'n') else batch.n_pts,
            i['in_training']: training,
        }
        for p, v in zip(i['mean_matrix'], mean_matrix(batch, sparse=True)):
            d[p] = v
        if labels is not None:
            d[i['y']] = labels
        return d

    def bag_pool_layer(self, layer):
        mean_matrix = tf.SparseTensor(*self.inputs['mean_matrix'])
        return tf.sparse_tensor_dense_matmul(mean_matrix, layer)
