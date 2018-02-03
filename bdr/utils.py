from __future__ import division

from contextlib import contextmanager

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def loop_batches(feats, max_pts, max_bags=np.inf, shuffle=False, stack=False):
    '''
    Loop over feats, yielding subsets with no more than max_pts total points
    in them and no more than max_bags bags.
    '''
    if shuffle:
        feats = feats[np.random.permutation(len(feats))]  # doesn't copy data
    
    rest = feats
    while len(rest):
        pts_i = np.cumsum(rest.n_pts).searchsorted(max_pts)
        how_many = min(pts_i, max_bags)
        if how_many == 0:
            raise ValueError("Bag of size {} doesn't work with max_pts {}"
                             .format(rest.n_pts[0], max_pts))
        this = rest[:how_many]
        rest = rest[how_many:]
        if stack:
            this.make_stacked()
        yield this


def get_median_sqdist(feats, n_sub=1000):
    feats.make_stacked()
    all_Xs = feats.stacked_features
    N = all_Xs.shape[0]
    sub = all_Xs[np.random.choice(N, min(n_sub, N), replace=False)]
    D2 = euclidean_distances(sub, squared=True)
    return np.median(D2[np.triu_indices_from(D2, k=1)], overwrite_input=True)


@contextmanager
def tf_session(n_cpus=1, config_args={}, **kwargs):
    import tensorflow as tf
    config = tf.ConfigProto(intra_op_parallelism_threads=n_cpus,
                            inter_op_parallelism_threads=n_cpus, **config_args)
    with tf.Session(config=config) as sess:
        yield sess
