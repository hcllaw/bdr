from __future__ import division
from functools import partial

import numpy as np
from six.moves import xrange
import tensorflow as tf

from .base import mean_matrix, sparse_matrix_placeholders
from .radial import _rbf_kernel


class BayesianLinearRegression(object):
    def __init__(self, in_dim, feat_dim, init_obs_var=1, init_weight_var=1,
                 dtype=tf.float32):
        self.dtype = dtype
        self.in_dim = in_dim
        self.feat_dim = feat_dim

        pl = partial(tf.placeholder, dtype)
        self.inputs = i = {
            'train_X': pl([None, in_dim]),  # all bags stacked up
            # mean_matrix is n_bags x n_pts; these make a SparseTensor
            'train_mean_matrix': sparse_matrix_placeholders(dtype),
            'train_y': pl([None]),  # one per bag
            'in_training': tf.placeholder(tf.bool, shape=[]),  # for batch norm
            'test_X': pl([None, in_dim]),
            'test_mean_matrix': sparse_matrix_placeholders(dtype),
        }

        # Model parameters
        self.params = params = getattr(self, 'params', {})
        const = partial(tf.constant, dtype=dtype)

        params['log_obs_var'] = tf.Variable(const(np.log(init_obs_var)))
        self.obs_var = tf.exp(params['log_obs_var'])

        params['log_weight_var'] = tf.Variable(const(np.log(init_weight_var)))
        self.weight_var = tf.exp(params['log_weight_var'])

        # Build graph
        train_feats = self.feat_transform(
            i['train_X'], tf.SparseTensor(*i['train_mean_matrix']))
        m_n, L_S_w = self.weight_posteriors(train_feats, i['train_y'])

        self.train_pred_mean, self.train_pred_var = self.predictions(
            m_n, L_S_w, train_feats)
        self.train_mse = tf.reduce_mean(tf.square(
            i['train_y'] - self.train_pred_mean))
        self.loss = self.model_nll(
            m_n, L_S_w, train_feats, i['train_y'], self.train_pred_mean)

        test_feats = self.feat_transform(
            i['test_X'], tf.SparseTensor(*i['test_mean_matrix']))
        self.pred_mean, self.pred_var = self.predictions(
            m_n, L_S_w, test_feats)

    def feat_transform(self, X, mean_matrix):
        raise NotImplementedError("Need to use a subclass for actual features.")

    def weight_posteriors(self, train_feats, train_y):
        # Bayesian linear regression steps
        # See e.g. http://www.utstat.utoronto.ca/~radford/sta414.S11/week4a.pdf
        prior_prec_w = tf.eye(self.feat_dim, dtype=self.dtype) / self.weight_var

        train_feat_cov = tf.matmul(train_feats, train_feats, transpose_a=True)
        train_feat_y = tf.matmul(train_feats, tf.expand_dims(train_y, 1),
                                 transpose_a=True)
        posterior_prec_w = prior_prec_w + train_feat_cov / self.obs_var

        posterior_prec_w_chol = tf.cholesky(posterior_prec_w)
        posterior_mean_w = tf.cholesky_solve(
            posterior_prec_w_chol, train_feat_y / self.obs_var)
        # ^ since we have prior_mean_w = 0

        return posterior_mean_w, posterior_prec_w_chol

    def predictions(self, posterior_mean_w, posterior_prec_w_chol, feats):
        pred_mean = tf.squeeze(tf.matmul(feats, posterior_mean_w), 1)

        # predictive var: x^T S x + sigma^2
        #               = x^T (L L^T)^-1 x + sigma^2
        #               = (x^T L^-T) (L^-1 x) + sigma^2
        #               = || L^-1 x ||^2 + sigma^2
        a = tf.matrix_triangular_solve(posterior_prec_w_chol,
                                       tf.transpose(feats))  # p x N
        pred_var = tf.reduce_sum(tf.square(a), 0) + self.obs_var
        return pred_mean, pred_var

    def model_nll(self, posterior_mean_w, posterior_prec_w_chol,
                  feats, y, preds):
        N = tf.cast(tf.shape(y)[0], self.dtype)
        p = tf.constant(self.feat_dim, self.dtype)

        t0 = N * (np.log(2 * np.pi) / 2)

        t1 = N / 2 * self.params['log_obs_var']

        log_det_prior_cov_w = p * self.params['log_weight_var']
        log_det_posterior_cov_w = -2 * tf.log(
            tf.reduce_prod(tf.matrix_diag_part(posterior_prec_w_chol)))
        t2 = .5 * (log_det_prior_cov_w - log_det_posterior_cov_w)

        t3 = tf.nn.l2_loss(y - preds) / self.obs_var

        t4 = tf.nn.l2_loss(posterior_mean_w) / self.weight_var

        return t0 + t1 + t2 + t3 + t4

    def fit(self, sess, train, val, checkpoint_path,
            lr=0.01, first_early_stop_epoch=None, max_epochs=1000,
            display_every=1):
        if first_early_stop_epoch is None:
            first_early_stop_epoch = max_epochs // 3

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # Make sure we do update_ops, e.g. for batch norm, before stepping
            optimize_step = tf.train.AdamOptimizer(lr).minimize(self.loss)

        cur_min = np.inf  # used for early stopping
        countdown = np.inf

        saver = tf.train.Saver(max_to_keep=1)
        sess.run(tf.global_variables_initializer())

        i = self.inputs
        train.make_stacked()
        self.train_X = train.stacked_features
        self.train_y = train.y
        self.train_mean_matrix = mean_matrix(train, sparse=True)
        train_d = {
            i['train_X']: self.train_X,
            i['train_y']: self.train_y,
            i['in_training']: True,
        }
        for p, v in zip(i['train_mean_matrix'], self.train_mean_matrix):
            train_d[p] = v

        val.make_stacked()
        val_d = {
            i['train_X']: val.stacked_features,
            i['train_y']: val.y,
            i['in_training']: False,
        }
        for p, v in zip(i['train_mean_matrix'], mean_matrix(val, sparse=True)):
            val_d[p] = v

        print("Training up to {} epochs; considering early stopping from {}."
              .format(max_epochs, first_early_stop_epoch))
        print("First columns: epoch, prior observation noise std, "
              "prior regression weight std.")
        # Training loop
        epoch = -np.inf
        for epoch in xrange(max_epochs):
            _, train_loss, train_mse, obs_std, weight_std = sess.run(
                [optimize_step, self.loss, self.train_mse,
                 tf.sqrt(self.obs_var), tf.sqrt(self.weight_var)],
                feed_dict=train_d)
            train_loss /= len(train)

            if epoch >= first_early_stop_epoch or epoch % display_every == 0:
                val_loss, val_mse = sess.run(
                    [self.loss, self.train_mse], feed_dict=val_d)
                val_loss /= len(val)

                if val_loss <= cur_min and epoch >= first_early_stop_epoch:
                    countdown = 10
                    cur_min = val_loss
                    save_path = saver.save(sess, checkpoint_path)
                    # TODO: this saving is a huge portion of training time.
                    #       figure out another way to do it...
                    best_epoch = epoch
                else:
                    countdown -= 1

            if epoch % display_every == 0:
                s = ("{: 4d}: {:6.3f} {:6.3f}; "
                     "train NLL = {:8.4f}  MSE = {:8.5f};  "
                     "val NLL = {:8.4f}  MSE = {:8.5f}")
                print(s.format(epoch, obs_std, weight_std,
                               train_loss, train_mse, val_loss, val_mse))

            if epoch >= first_early_stop_epoch and countdown <= 0:
                break

        if epoch >= first_early_stop_epoch:
            print(("Stopping at epoch {} with val mean NLL {:.8}\n"
                   "Using model from epoch {} with val mean NLL {:.8}").format(
                       epoch, val_loss, best_epoch, cur_min))
            saver.restore(sess, save_path)
        else:
            print("Using final model.")

    def predict(self, sess, test):
        test.make_stacked()
        i = self.inputs
        d = {
            i['train_X']: self.train_X,
            i['train_y']: self.train_y,
            i['in_training']: False,
            i['test_X']: test.stacked_features,
        }
        for p, v in zip(i['train_mean_matrix'], self.train_mean_matrix):
            d[p] = v
        for p, v in zip(i['test_mean_matrix'], mean_matrix(test, sparse=True)):
            d[p] = v
        return sess.run([self.pred_mean, self.pred_var], feed_dict=d)


class RadialBLR(BayesianLinearRegression):
    def __init__(self, landmarks, bw, use_batch_norm=False, opt_landmarks=True,
                 dtype=tf.float32, **kwargs):
        self.use_batch_norm = use_batch_norm

        # Model parameters
        const = partial(tf.constant, dtype=dtype)
        self.params = {
            'landmarks': tf.Variable(const(landmarks), trainable=opt_landmarks),
            'log_bw': tf.Variable(const(np.log(bw)), trainable=opt_landmarks),
        }

        super(RadialBLR, self).__init__(
            in_dim=landmarks.shape[1], feat_dim=landmarks.shape[0],
            dtype=dtype, **kwargs)

    def feat_transform(self, X, mean_matrix):
        # Compute kernels to landmark points: shape (n_X, n_land)
        kernel_layer = _rbf_kernel(
            X, self.params['landmarks'], self.params['log_bw'])

        # Pool bags: shape (n_bags, n_land)
        layer_pool = tf.sparse_tensor_dense_matmul(mean_matrix, kernel_layer)

        # Batch normalization, maybe
        if self.use_batch_norm:
            # Think something might be up with this, behaving strangely...
            return tf.contrib.layers.batch_norm(
                layer_pool, center=True, scale=False,
                # no need to scale since next layer is linear
                is_training=self.inputs['in_training'], scope='bn')
        else:
            return layer_pool


class LinearBLR(BayesianLinearRegression):
    def __init__(self, dim, **kwargs):
        super(LinearBLR, self).__init__(in_dim=dim, feat_dim=dim, **kwargs)

    def feat_transform(self, X, mean_matrix):
        return tf.sparse_tensor_dense_matmul(mean_matrix, X)
