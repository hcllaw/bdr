#!/usr/bin/env python
from __future__ import division, print_function
from functools import partial
import os
import sys

import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics.pairwise import rbf_kernel
import tensorflow as tf

from bdr import Features
from bdr.networks.blr import LinearBLR, RadialBLR
from bdr.utils import get_median_sqdist, tf_session

sys.path.append(os.path.dirname(__file__))
from train_test import parse_args, generate_data, pick_landmarks


def _rest_of_args(subparser):
    network = subparser.add_argument_group("Network parameters")
    n = partial(network.add_argument, help="Default %(default)s.")
    n('--type', choices=['radial', 'linear', 'radial-precomp'],
      default='radial')
    n('--n-landmarks', type=int, default=30)
    n('--opt-landmarks', action='store_true', default=True)
    n('--no-opt-landmarks', action='store_false', dest='opt_landmarks')
    n('--kmeans-landmarks', action='store_true', default=False)
    n('--bw-scale', type=float, default=1)
    n('--init-weight-var', type=float, default=1)
    n('--init-obs-var-mult', type=float, default=1)
    n('--landmark-seed', type=int, default=1)

    train = subparser.add_argument_group("Training parameters")
    t = partial(train.add_argument, help="Default %(default)s.")
    t('--max-epochs', type=int, default=1000)
    train.add_argument('--first-early-stop-epoch', type=int,
                       help="Default: MAX_EPOCHS / 3.")
    t('--learning-rate', '--lr', type=float, default=.01)
    t('--dtype-double', action='store_true', default=False)
    t('--dtype-single', action='store_false', dest='dtype_double')

    io = subparser.add_argument_group("I/O parameters")
    io.add_argument('out_dir')
    io.add_argument('--n-cpus', type=int, default=8)


def main():
    args = parse_args(_rest_of_args)

    print("Loading data...")
    train, estop, val, test = generate_data(args)

    print("Building network...")
    kw = {
        'init_obs_var': np.var(train.y) * args.init_obs_var_mult,
        'init_weight_var': args.init_weight_var,
        'dtype': tf.float64 if args.dtype_double else tf.float32,
    }

    if args.type == 'radial':
        cls = RadialBLR
        kw['bw'] = np.sqrt(get_median_sqdist(train) / 2) * args.bw_scale
        kw['landmarks'] = pick_landmarks(args, train)
        kw['opt_landmarks'] = args.opt_landmarks
        kw['use_batch_norm'] = False
    elif args.type == 'radial-precomp':
        cls = LinearBLR
        bw = np.sqrt(get_median_sqdist(train) / 2) * args.bw_scale
        gamma = 1 / (2 * bw**2)
        landmarks = pick_landmarks(args, train)
        kw['dim'] = landmarks.shape[0]

        def transform(feats):
            feats.make_stacked()
            x = rbf_kernel(
                feats.stacked_features, landmarks, gamma=gamma)
            return Features(x, feats.n_pts, **feats.meta)
        train = transform(train)
        estop = transform(estop)
        val = transform(val)
        test = transform(test)

    elif args.type == 'linear':
        cls = LinearBLR
        kw['dim'] = train.dim
    else:
        raise ValueError("unknown type {}".format(args.type))

    if args.type in {'linear', 'radial-precomp'}:
        # only care about the mean
        train = train.means()
        estop = estop.means()
        val = val.means()
        test = test.means()

    blr = cls(**kw)
    d = {'args': args}

    with tf_session(n_cpus=args.n_cpus) as sess:
        print("Fitting network...")
        blr.fit(sess, train, estop,
                checkpoint_path=os.path.join(args.out_dir, 'checkpoints/model'),
                lr=args.learning_rate, max_epochs=args.max_epochs,
                first_early_stop_epoch=args.first_early_stop_epoch)

        print("Evaluating...")
        for name, ds in [('val', val), ('test', test)]:
            print()
            preds, preds_var = blr.predict(sess, ds)

            d[name + '_y'] = y = ds.y
            d[name + '_preds'] = preds
            d[name + '_preds_var'] = preds_var

            d[name + '_mse'] = mse = mean_squared_error(y, preds)
            print('{} MSE: {}'.format(name, mse))

            d[name + '_r2'] = r2 = r2_score(y, preds)
            print('{} R2: {}'.format(name, r2))

            liks = stats.norm.pdf(y, preds, np.sqrt(preds_var))
            d[name + '_nll'] = nll = -np.log(liks).mean()
            print('{} NLL: {}'.format(name, nll))

            cdfs = stats.norm.cdf(y, preds, np.sqrt(preds_var))
            d[name + '_coverage'] = cov = np.mean((cdfs > .025) & (cdfs < .975))
            print('{} coverage at 95%: {:.1%}'.format(name, cov))

    np.savez(os.path.join(args.out_dir, 'results.npz'), **d)


if __name__ == '__main__':
    main()
