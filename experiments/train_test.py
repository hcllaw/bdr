#!/usr/bin/env python
from __future__ import division, print_function
import argparse
from functools import partial
import multiprocessing
import os
import sys

import numpy as np
from scipy import stats
from six.moves import xrange
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import ShuffleSplit, GroupShuffleSplit
from sklearn.utils import check_random_state
import tensorflow as tf

# TO FIX LEON 
from bdr import networks
from bdr.train import eval_network, train_network
from bdr.utils import get_median_sqdist, tf_session

def get_adder(g):
    def f(*args, **kwargs):
        kwargs.setdefault('help', "Default %(default)s.")
        return g.add_argument(*args, **kwargs)
    return f


network_types = {
    'radial': networks.build_radial_net,
    'shrinkage': networks.build_shrinkage_net,
    'shrinkage_novar': networks.build_shrinkage_novar_net,
    'fourier': networks.build_fourier_net,
}


def _add_args(subparser):
    network = subparser.add_argument_group("Network parameters")
    n = get_adder(network)
    n('--type', '-n', required=True, choices=network_types)
    n('--n-landmarks', type=int, default=30)
    n('--opt-landmarks', action='store_true', default=True)
    n('--no-opt-landmarks', action='store_false', dest='opt_landmarks')
    n('--kmeans-landmarks', action='store_true', default=False)
    n('--landmark-seed', type=int, default=1)
    n('--bw-scale', type=float, default=1)
    n('--n-freqs', type=int, default=64)
    n('--reg-out', type=float, default=0)
    n('--reg-out-bias', type=float, default=0)
    n('--reg-obs-var', type=float, default=0)
    n('--reg-freqs', type=float, default=0)
    n('--scale-reg-by-n', action='store_true', default=False)
    n('--init-from-ridge', action='store_true', default=False)
    n('--init-prior-feat-var', type=float, default=1)
    n('--opt-prior-feat-var', action='store_true', default=True)
    n('--fix-prior-feat-var', action='store_false', dest='opt_prior_feat_var')
    n('--shrink-towards-mean', action='store_true', default=False)
    n('--use-empirical-cov', action='store_true', default=False)
    n('--empirical-cov-add-diag', type=float, default=0)
    n('--use-alpha-reg', action='store_true', default=False,
      help='Regularize the regression with an RKHS norm, instead of an L2 norm '
           'on the weights. Default %(default)s.')

    g = get_adder(network.add_mutually_exclusive_group())
    g('--use-cholesky', action='store_true', default=True)
    g('--no-cholesky', action='store_false', dest='use_cholesky')

    G = network.add_mutually_exclusive_group()
    g = get_adder(G)
    g('--init-obs-var', type=float, default=None)
    g('--init-obs-var-mult', type=float, default=1)

    G = network.add_mutually_exclusive_group()
    g = get_adder(G)
    g('--use-real-R', action='store_true', default=False)
    g('--use-rbf-R', action='store_false', dest='use_real_R')
    n('--init-prior-measure-var', type=float, default=1)
    n('--init-R-scale', type=float, default=1)
    n('--opt-R-scale', action='store_true', default=False)
    n('--rbf-R-bw-scale', type=float, default=np.sqrt(2),
      help="Scale the bandwidth of the rbf R (default sqrt(2)).")

    train = subparser.add_argument_group("Training parameters")
    t = get_adder(train)
    t('--max-epochs', type=int, default=1000)
    t('--first-early-stop-epoch', type=int, help="Default: MAX_EPOCHS / 3.")
    int_inf = lambda x: np.inf if x.lower() in {'inf', 'none'} else int(x)
    t('--batch-pts', type=int_inf, default='inf')
    t('--batch-bags', type=int_inf, default=30)
    t('--eval-batch-pts', type=int_inf, default='inf')
    t('--eval-batch-bags', type=int_inf, default=100)
    t('--learning-rate', '--lr', type=float, default=.01)
    t('--dtype-double', action='store_true', default=False)
    t('--dtype-single', action='store_false', dest='dtype_double')
    t('--optimizer', choices=['adam', 'sgd'], default='adam')

    io = subparser.add_argument_group("I/O parameters")
    i = get_adder(io)
    io.add_argument('out_dir')
    i('--n-cpus', type=int, default=min(8, multiprocessing.cpu_count()))


def make_parser(rest_of_args=_add_args):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="The dataset to run on")
    # Subparser chosen by the first argument of your parser
    def add_subparser(name, **kwargs):
        subparser = subparsers.add_parser(name, **kwargs)
        subparser.set_defaults(dataset=name)
        data = subparser.add_argument_group('Data parameters')
        rest_of_args(subparser)
        return data, get_adder(data)

    chi2, d = add_subparser('chi2')
    d('--n-train', type=int, default=1000)
    d('--n-estop', type=int, default=500)
    d('--n-val',   type=int, default=500)
    d('--n-test',  type=int, default=1000)
    d('--dim', '-d', type=int, default=5)
    d('--data-seed', type=int, default=np.random.randint(2**32))
    d('--size-type', choices=['uniform', 'neg-binom', 'manual', 'special'],
      default='uniform', help='special for fixed bag size of 1000')
    d('--min-size', type=int, default=20)
    d('--max-size', type=int, default=100)
    d('--size-mean', type=int, default=20)
    d('--size-std', type=int, default=10)
    d('--min-df', type=float, default=4)
    d('--max-df', type=float, default=8)
    d('--noise-std', type=float, default=0)

    def add_split_args(g):
        a = g.add_argument
        a('--test-size', type=float, default=.2,
          help="Number or portion of overall data to use for testing "
               "(default %(default)s).")
        a('--trainval-size', type=float, default=None,
          help="Number or portion of overall data to use for training, "
               "early-stopping, and validation together "
               "(default: complement of --test-size).")
        a('--val-size', type=float, default=.1875,
          help="Number or portion of non-test data to use for validation "
               "(default %(default)s).")
        a('--train-estop-size', type=float, default=None,
          help="Number or portion of non-test data to use for training and "
               "early stopping together (default: complement of "
               "--val-size).")
        a('--estop-size', type=float, default=0.2308,
          help="Number or portion of train-estop data to use for "
               "early stopping (default %(default)s).")
        a('--train-size', type=float, default=None,
          help="Number or portion of train-estop data to use for training "
               "(default: complement of --estop-size).")
        a('--split-seed', type=int, default=np.random.randint(2**32),
          help="Seed for the split process (default: random).")

    faces, d = add_subparser('imdb_faces')
    add_split_args(faces)

    return parser


def check_output_dir(dirname, parser, make_checkpoints=False):
    checkpoints = os.path.join(dirname, 'checkpoints')
    if os.path.exists(dirname):
        files = set(os.listdir(dirname)) - {'output'}
        if 'checkpoints' in files:
            if not os.listdir(checkpoints):
                files.discard('checkpoints')
        if files:
            parser.error(("Output directory {} exists, change the name or delete it.")
                         .format(dirname))
    else:
        os.makedirs(dirname)

    if make_checkpoints and not os.path.isdir(checkpoints):
        os.makedirs(checkpoints)


def parse_args(rest_of_args=_add_args):


def _split_feats(args, feats, labels=None, groups=None):
    if groups is None:
        ss = ShuffleSplit
    else:
        ss = GroupShuffleSplit

    rs = check_random_state(args.split_seed)
    test_splitter = ss(
        1, train_size=args.trainval_size, test_size=args.test_size,
        random_state=rs)
    (trainval, test), = test_splitter.split(feats, labels, groups)

    val_splitter = ss(
        1, train_size=args.train_estop_size, test_size=args.val_size,
        random_state=rs)
    X_v = feats[trainval]
    y_v = None if labels is None else labels[trainval]
    g_v = None if groups is None else groups[trainval]
    (train_estop, val), = val_splitter.split(X_v, y_v, g_v)

    estop_splitter = ss(
        1, train_size=args.train_size, test_size=args.estop_size,
        random_state=rs)
    X = X_v[train_estop]
    y = None if labels is None else y_v[train_estop]
    g = None if groups is None else g_v[train_estop]
    (train, estop), = estop_splitter.split(X, y, g)

    return X[train], X[estop], X_v[val], feats[test]


def generate_data(args):
    if args.dataset == 'chi2':
        from bdr.data.toy import generate_chisq
        d = dict(
            bag_sizes=[args.min_size, args.max_size],
            dim=args.dim,
            df_range=[args.min_df, args.max_df],
            noise=args.noise_std,
        )
        if args.size_type == 'uniform':
            d['size_type'] = 'uniform'
            d['bag_sizes'] = [args.min_size, args.max_size]
        elif args.size_type == 'neg-binom':
            d['size_type'] = 'neg-binom'
            d['bag_sizes'] = [args.size_mean, args.size_std]
        elif args.size_type == 'manual':
            d['size_type'] = 'manual'
            d['bag_sizes'] = [args.min_size * .01, .25, .25,
                              args.max_size * .01]
        elif args.size_type == 'special':
            d['size_type'] = 'manual'
            d['bag_sizes'] = [0.0, 0.0, 0.0, 1.0]
        else:
            raise ValueError("unknown size_type {}".format(args.size_type))
        make = partial(generate_chisq, **d)
        rs = check_random_state(args.data_seed)
        args.train_seed, args.estop_val_seed, args.optval_seed, args.test_seed \
            = rs.randint(2**32, size=4)
        train = make(args.n_train, seed=args.train_seed)
        estop = make(args.n_estop, seed=args.estop_val_seed)
        val   = make(args.n_val,   seed=args.optval_seed)
        test  = make(args.n_test,  seed=args.test_seed)
        return train, estop, val, test
    elif args.dataset == 'imdb_faces':
        from bdr.data.imdb_faces import load_faces, get_emp_cov
        feats = load_faces()
        train, estop, val, test = _split_feats(args, feats, feats.y)
        train.emp_cov_info = get_emp_cov()  # hack :|
        return train, estop, val, test
    else:
        raise ValueError("unknown dataset {}".format(args.dataset))


def pick_landmarks(args, train):
    train.make_stacked()
    rs = check_random_state(args.landmark_seed)
    if args.kmeans_landmarks:
        kmeans = KMeans(n_clusters=args.n_landmarks, random_state=rs, n_jobs=1)
        kmeans.fit(train.stacked_features)
        return kmeans.cluster_centers_
    else:
        w = rs.choice(train.total_points, args.n_landmarks, replace=False)
        return train.stacked_features[w]


def make_network(args, train):
    kw = dict(
        in_dim=train.dim,
        reg_out=args.reg_out,
        reg_out_bias=args.reg_out_bias,
        scale_reg_by_n=args.scale_reg_by_n,
        dtype=tf.float64 if args.dtype_double else tf.float32,
    )

    if args.type in {'radial', 'shrinkage', 'shrinkage_novar', 'fourier'}:
        kw['bw'] = bw = np.sqrt(get_median_sqdist(train) / 2) * args.bw_scale

    if 'shrinkage' or 'shrinkage_novar' in args.type:
        need_means = args.shrink_towards_mean or args.init_from_ridge
        need_all_feats = args.use_empirical_cov
    else:
        need_means = args.init_from_ridge
        need_all_feats = False

    # set up basic arguments, get featurization if we need it
    if args.type == 'fourier':
        kw['n_freqs'] = n_freqs = args.n_freqs
        kw['reg_freqs'] = args.reg_freqs
        kw['init_freqs'] = freqs = np.random.normal(
            scale=1/bw, size=(train.dim, args.n_freqs))

        if need_all_feats:
            train.make_stacked()
            angles = np.dot(train.stacked_features, freqs)
            train_feats = Features(
                np.concat([np.sin(angles), np.cos(angles)], 1), train.n_pts)
            if need_means:
                train_means = train_feats.means().stacked_features
        elif need_means:
            train_means = np.empty((len(train), 2 * n_freqs))
            for i, bag in enumerate(train):
                angles = np.dot(bag, freqs)
                train_means[i, :n_freqs] = np.sin(angles).mean(axis=0)
                train_means[i, n_freqs:] = np.cos(angles).mean(axis=0)

    elif args.type in {'radial', 'shrinkage', 'shrinkage_novar'}:
        kw['landmarks'] = landmarks = pick_landmarks(args, train)
        kw['opt_landmarks'] = args.opt_landmarks

        get_f = partial(rbf_kernel, Y=landmarks, gamma=1 / (2 * bw**2))
        if need_all_feats:
            train.make_stacked()
            train_feats = Features(get_f(train.stacked_features), train.n_pts)
            if need_means:
                train_means = train_feats.means().stacked_features
        elif need_means:
            train_means = np.empty((len(train), landmarks.shape[0]))
            for i, bag in enumerate(train):
                train_means[i] = get_f(bag).mean(axis=0)

    if args.init_from_ridge:
        print("Fitting ridge...")
        ridge = Ridge(alpha=2 * args.reg_out * len(train),
                      solver='saga', tol=0.1, max_iter=500)
        # Ridge mins  ||y - X w - b||^2 + alpha ||w||^2
        # we min  1/n ||y - X w - b||^2 + reg_out/2 ||w||^2
        # (at least in radial...shrinkage a bit different, but w/e)
        ridge.fit(train_means, train.y)
        init_mse = mean_squared_error(train.y, ridge.predict(train_means))
        print("Ridge MSE: {:.4f}".format(init_mse))
        kw['init_out'] = ridge.coef_
        kw['init_out_bias'] = ridge.intercept_

    if args.type in {'shrinkage', 'shrinkage_novar'}:
        if args.type != 'shrinkage_novar':
            kw['reg_obs_var'] = args.reg_obs_var
            if args.init_obs_var is not None:
                kw['init_obs_var'] = args.init_obs_var
            elif args.init_from_ridge:
                kw['init_obs_var'] = init_mse
            else:
                kw['init_obs_var'] = args.init_obs_var_mult * np.var(train.y)

        if args.type in {'shrinkage', 'shrinkage_novar'}:
            kw['use_cholesky'] = args.use_cholesky

        if args.shrink_towards_mean:
            kw['shrink_towards'] = train_means.mean(axis=0)

        if args.use_empirical_cov:
            if hasattr(train_feats, 'emp_cov_info'):
                cov, cov_eigvals, cov_eigvecs = train.emp_cov_info
            else:
                covs = []
                for bag in train_feats:
                    if bag.shape[0] > 1:
                        covs.append(np.cov(bag, rowvar=False))
                cov = np.mean(covs, axis=0)

            if args.empirical_cov_add_diag:
                d = cov.shape[0]
                cov[xrange(d), xrange(d)] += args.empirical_cov_add_diag

            kw['fixed_cov_matrix'] = cov
        else:
            kw['init_tau_sq'] = args.init_prior_feat_var
            kw['opt_tau_sq'] = args.opt_prior_feat_var

        if args.type in {'shrinkage', 'shrinkage_novar'}:
            kw['use_real_R'] = args.use_real_R
            kw['init_eta_sq'] = args.init_prior_measure_var
            kw['rbf_R_bw_scale'] = args.rbf_R_bw_scale
            kw['use_alpha_reg'] = args.use_alpha_reg

        kw['init_R_scale'] = args.init_R_scale
        kw['opt_R_scale'] = args.opt_R_scale

    return network_types[args.type](**kw)


def train_net(sess, args, net, train, val):
    optimizer = {
        'adam': tf.train.AdamOptimizer,
        'sgd': tf.train.GradientDescentOptimizer,
    }[args.optimizer]
    train_network(sess, net, train, val,
                  os.path.join(args.out_dir, 'checkpoints/model'),
                  batch_pts=args.batch_pts, batch_bags=args.batch_bags,
                  eval_batch_pts=args.eval_batch_pts,
                  eval_batch_bags=args.eval_batch_bags,
                  max_epochs=args.max_epochs,
                  first_early_stop_epoch=args.first_early_stop_epoch,
                  optimizer=optimizer,
                  lr=args.learning_rate)

def eval_net(sess, args, net, test, do_var=False):
    return eval_network(sess, net, test,
                        batch_pts=args.eval_batch_pts,
                        batch_bags=args.eval_batch_bags,
                        do_var=do_var)

def main():
    args = parse_args()
    print("Loading data...")
    train, estop, val, test = generate_data(args)
    print("Constructing network...")
    net = make_network(args, train)
    if getattr(net, 'wants_means_only', False):
        print("Precomputing features...")
        if hasattr(net, 'landmarks'):
            def transform(feats):
                feats.make_stacked()
                x = rbf_kernel(
                    feats.stacked_features, net.landmarks, gamma=net.gamma)
                return Features(x, feats.n_pts, **feats.meta)
            train = transform(train)
            estop = transform(estop)
            val = transform(val)
            test = transform(test)
        train = train.means()
        estop = estop.means()
        val = val.means()
        test = test.means()
    do_var = hasattr(net, 'output_var')

    d = {'args': args}

    with tf_session(n_cpus=args.n_cpus) as sess:
        train_net(sess, args, net, train, estop)

        for name, ds in [('val', val), ('test', test)]:
            print()
            preds = eval_net(sess, args, net, ds, do_var=do_var)
            if do_var:
                preds, preds_var = preds
                d[name + '_preds_var'] = preds_var

            d[name + '_y'] = y = ds.y
            d[name + '_preds'] = preds

            d[name + '_mse'] = mse = mean_squared_error(y, preds)
            print('{} MSE: {}'.format(name, mse))

            d[name + '_r2'] = r2 = r2_score(y, preds)
            print('{} R2: {}'.format(name, r2))

            if do_var:
                liks = stats.norm.pdf(y, preds, np.sqrt(preds_var))
                d[name + '_nll'] = nll = -np.log(liks).mean()
                print('{} NLL: {}'.format(name, nll))

                cdfs = stats.norm.cdf(y, preds, np.sqrt(preds_var))
                coverage = np.mean((cdfs > .025) & (cdfs < .975))
                d[name + '_coverage'] = coverage
                print('{} coverage at 95%: {:.1%}'.format(name, coverage))

    np.savez(os.path.join(args.out_dir, 'results.npz'), **d)


if __name__ == '__main__':
    main()
