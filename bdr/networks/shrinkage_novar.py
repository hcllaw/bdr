from __future__ import division

import numpy as np
import tensorflow as tf

from .base import Network


def build_shrinkage_novar_net(in_dim, landmarks, bw, reg_out,
                        reg_out_bias=0, scale_reg_by_n=False,
                        dtype=tf.float32, use_batch_norm=False,
                        init_out=None, init_out_bias=None,
                        init_tau_sq=1, init_eta_sq=1,
                        opt_tau_sq=True,
                        opt_landmarks=True,
                        shrink_towards=0, opt_shrinkage_mean=False,
                        use_real_R=True,
                        init_R_scale=1, opt_R_scale=False,
                        rbf_R_bw_scale=np.sqrt(2),
                        fixed_cov_matrix=None, use_alpha_reg=False,
                        precompute_feats=None, use_cholesky=True):
    n_land = landmarks.shape[0]

    if precompute_feats is None:
        precompute_feats = not opt_landmarks
    elif precompute_feats:
        assert not opt_landmarks

    net = Network(n_land if precompute_feats else in_dim, n_land, dtype=dtype)
    inputs = net.inputs
    params = net.params

    if precompute_feats:
        net.wants_means_only = True
        net.landmarks = landmarks  # save outside of tensorflow for convenience
        net.gamma = 1 / (2 * bw**2)

    # Model parameters
    params['landmarks'] = tf.Variable(
        tf.constant(landmarks, dtype=dtype), trainable=opt_landmarks)
    params['log_bw'] = tf.Variable(
        tf.constant(np.log(bw), dtype=dtype), trainable=opt_landmarks)

    params['shrink_towards'] = tf.Variable(
        tf.constant(shrink_towards, dtype=dtype, shape=[n_land]),
        trainable=opt_shrinkage_mean)
    params['log_R_scale'] = tf.Variable(
        tf.constant(np.log(init_R_scale), dtype=dtype), trainable=opt_R_scale)

    if init_out is None:
        out = tf.random_normal([n_land], dtype=dtype)
    else:
        assert np.size(init_out) == n_land
        out = tf.constant(np.resize(init_out, [n_land]), dtype=dtype)
    params['out'] = tf.Variable(out)

    if init_out_bias is None:
        out_bias = tf.random_normal([1], dtype=dtype)
    else:
        out_bias = tf.constant(init_out_bias, shape=(), dtype=dtype)
    params['out_bias'] = tf.Variable(out_bias)

    if fixed_cov_matrix is not None:
        params['Sigma'] = tf.Variable(
            tf.constant(fixed_cov_matrix, dtype=dtype), trainable=False)
    else:
        params['log_tau_sq'] = tf.Variable(
            tf.constant(np.log(init_tau_sq), dtype=dtype), trainable=opt_tau_sq)
        tau_sq = tf.exp(params['log_tau_sq'])
    if use_real_R:
        params['log_eta_sq'] = tf.Variable(
            tf.constant(np.log(init_eta_sq), dtype=dtype))
        eta_sq = tf.exp(params['log_eta_sq'])

    # compute kernel stuff, reusing some temp values

    Y = params['landmarks']
    YY = tf.matmul(Y, Y, transpose_b=True)  # Y Y^T, Y is matrix of landmarks
    Y_sqnorms = tf.diag_part(YY)  # The norm square of Y

    gamma = 1 / (2 * tf.exp(2 * params['log_bw']))  # = 1 / (2 bw^2)
    if not precompute_feats:
        X = inputs['X']
        XY = tf.matmul(X, Y, transpose_b=True)
        X_sqnorms_s = tf.reduce_sum(tf.square(X), 1, keep_dims=True)
        Y_sqnorms_s = tf.expand_dims(Y_sqnorms, 0)
        kernel_layer = tf.exp(-gamma * (-2 * XY + X_sqnorms_s + Y_sqnorms_s))
    else:
        kernel_layer = inputs['X']

    # Added by Leon, for regularisation parameter of alpha^T K alpha
    if use_alpha_reg:
        net.K = K = tf.exp(-gamma * (
            -2 * YY + tf.expand_dims(Y_sqnorms, 1)
                    + tf.expand_dims(Y_sqnorms, 0)))

    # R matrix, to be used later
    if use_real_R:
        bw_sq = tf.exp(2 * params['log_bw'])
        # R kernel from https://arxiv.org/pdf/1603.02160.pdf appendix A.3
        # Sigma_theta = bw^2 * I
        net.R = R = tf.exp(params['log_R_scale']) * (
            # |2 Sigma_theta^-1 + eta^-2 I_D|^-(1/2)
            #    = | (2 bw^-2 + eta^-2) I_D |^(-1/2)
            #    = (2 bw^-2 + \ta^-2)^(-D/2)
            (2 * np.pi / (2 / bw_sq + 1 / eta_sq)) ** (in_dim/2)
            * tf.exp(-1 / (4 * bw_sq) * (
                - 2 * YY
                + tf.expand_dims(Y_sqnorms, 1)  # Like broadcast it in two ways
                + tf.expand_dims(Y_sqnorms, 0)
            ))
            # - 1/8 (x + y)^T (1/2 Sigma_theta + eta^2 I)^-1 (x + y)
            # = -1/(8(bw^2/2 + eta^2)) (x + y)^T (x + y)
            # = -1/(8(bw^2/2 + eta^2)) (x^T x + y^T y + 2 x^T y)
            * tf.exp(-1 / (8 * (bw_sq/2 + eta_sq)) * (
                2 * YY
                + tf.expand_dims(Y_sqnorms, 1)
                + tf.expand_dims(Y_sqnorms, 0)
            ))
        )
    else:
        gamma /= rbf_R_bw_scale**2
        net.R = R = tf.exp(params['log_R_scale']) * tf.exp(-gamma * (
            -2 * YY + tf.expand_dims(Y_sqnorms, 1)
                    + tf.expand_dims(Y_sqnorms, 0)))

    # Pool bags: shape (n_bags, n_land)
    layer_pool = net.bag_pool_layer(kernel_layer)

    # Bayesian posterior stuff
    # We need (R + c I)^{-1} times a bunch of different matrices, for different
    # values of c. It'd probably be faster to eigendecompose R so that we can
    # quickly get the inverse for different values of c, but it seems like
    # tensorflow 1.0 doesn't have working gradients for eigendecompositions.
    # Note that it's not easy to get a Cholesky factor for different c:
    #   http://math.stackexchange.com/q/1276559/19147
    # Since there's no way to tell tf.matrix_solve that the matrix in question
    # is PSD, though, we might as well use cholesky_solve.

    # make R + tau^2 / n I matrices; shape will be (n_bags, n_land, n_land)
    # TODO: if n is repeated a lot, we could be smarter about this and avoid
    #       doing repeated choleskys...maybe worth it since it often is

    # compute chol(R + Sigma/n_i) matrices; shape (n_bags, n_land, n_land)
    if fixed_cov_matrix is not None:
        adds = (
            tf.expand_dims(params['Sigma'], 0)  # 1, n_land, n_land
            / tf.reshape(inputs['sizes'], [-1, 1, 1])  # n_bags, 1, 1
        )
    else:
        tau_sq = tf.exp(params['log_tau_sq'])
        adds = tf.matrix_diag(
            tf.tile(tf.expand_dims(tau_sq / inputs['sizes'], 1), [1, n_land]))

    to_inverts = tf.expand_dims(R, 0) + adds
    if use_cholesky:
        chols = tf.cholesky(to_inverts)

    b = tf.expand_dims(params['out'], 1)  # n_land, 1
    Rb = tf.matmul(R, b)  # n_land, 1

    # we need a version of Rb that's (n_bags, n_land, 1),
    # since cholesky_solve doesn't broadcast on first axis.
    # but tf.tile won't take a Dimension(None) argument.
    # so, here's a dumb dumb hack. there must be a better way to do this....
    reshaper = 0 * tf.expand_dims(tf.expand_dims(inputs['sizes'], 1), 2)
    Rb_broad = reshaper + tf.expand_dims(Rb, 0)  # n_bags, n_land, 1
    if use_cholesky:
        inv_Rb = tf.cholesky_solve(chols, Rb_broad)  # n_bags, n_land, 1
    else:
        inv_Rb = tf.matrix_solve(to_inverts, Rb_broad)

    # posterior mean is   b^T R (R + tau^2/n I)^-1 (mu_hat - m0) + b^T m0
    m0 = tf.expand_dims(params['shrink_towards'], 0)  # [1, n_land]
    mu_hats = layer_pool  # n_bags, n_land
    mu_hat_deltas = mu_hats - m0

    net.output = posterior_means = (
        params['out_bias']
        + tf.squeeze(tf.matmul(  # n_bags
            tf.expand_dims(mu_hat_deltas, 1),  # n_bags, 1, n_land
            inv_Rb                             # n_bags, n_land, 1
        ), [1, 2])
        + tf.squeeze(tf.matmul(m0, b), [0, 1])
    )

    net.early_stopper = .5 * tf.reduce_mean(
        tf.square(inputs['y'] - posterior_means) )

    if scale_reg_by_n:
        n = tf.cast(tf.squeeze(tf.shape(net.output), [0]), dtype)
        reg_out /= n
        reg_out_bias /= n
        reg_obs_var /= n
    if use_alpha_reg:  # a^T K a loss
        loss_weights = tf.squeeze(
            tf.matmul(tf.matmul(b, K, transpose_a=True), b))  # b is n_land, 1
    else:
        loss_weights = tf.nn.l2_loss(b)

    net.loss = (
        net.early_stopper
        + reg_out * loss_weights
        + reg_out_bias * tf.nn.l2_loss(params['out_bias'])
    )
    net.print_out_names = []
    net.print_out = []
    if fixed_cov_matrix is None and opt_tau_sq:
        net.print_out_names.append("tau^2")
        net.print_out.append(tau_sq)
    if use_real_R:
        net.print_out_names.append("eta^2")
        net.print_out.append(tf.exp(params['log_eta_sq']))
    if opt_R_scale:
        net.print_out_names.append("R scale")
        net.print_out.append(tf.exp(params['log_R_scale']))

    return net
