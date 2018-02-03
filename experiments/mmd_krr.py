#!/usr/bin/env python
from __future__ import division, print_function
from functools import partial
import os
import sys

import numpy as np
import progressbar as pb  # pip install progressbar2
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, r2_score

from mmd import rbf_mmd  # conda install -c dougal mmd   or   pip install mmd
from mmd.utils import show_progress
show_progress('mmd.mmd.progress')

from bdr.utils import get_median_sqdist

sys.path.append(os.path.dirname(__file__))
from train_test import make_parser, generate_data


default_l1_gamma_mults = np.array([1/64, 1/16, 1/4, 1, 4, 16])
default_l2_gamma_mults = np.array([1/64, 1/16, 1/4, 1, 4, 16])
default_alphas = np.array([1/128, 1/64, 1/16, 1/4, 1, 4])


def make_kernel(sqdists, gamma, add_one=True, out=None):
    if out is None:
        out = np.empty_like(sqdists)
    np.multiply(sqdists, -gamma, out=out)
    np.exp(out, out=out)
    out += 1
    return out


def main():
    args = parse_args()

    print("Loading data...")
    train, estop, val, test = generate_data(args)

    print("Getting training MMDs...")
    med_2 = get_median_sqdist(train)
    l1_gammas = args.l1_gamma_mults / med_2

    get_mmd_sq = partial(rbf_mmd, squared=True, n_jobs=args.n_jobs)

    train_mmd_sqs, train_mmk_sqnorms = get_mmd_sq(
        train, gammas=l1_gammas, ret_X_diag=True)

    print("Getting training-to-val MMDs...")
    val_mmd_sqs = get_mmd_sq(
        val, train, gammas=l1_gammas, Y_diag=train_mmk_sqnorms)

    if args.cache_mmd:
        np.savez(os.path.join(args.out_dir, 'mmd-cache.npz'),
                 train_mmd_sqs=train_mmd_sqs,
                 train_mmk_sqnorms=train_mmk_sqnorms,
                 val_mmd_sqs=val_mmd_sqs)

    print("Training models...")
    bias = train.y.mean()
    (model, (l1_gamma, l2_gamma, alpha), (l1_gamma_i, l2_gamma_i, alpha_i),
            scores, l2_gamma_scales) = \
        run_krr(train_mmd_sqs, train.y - bias, val_mmd_sqs, val.y - bias,
                l1_gammas=l1_gammas, l2_gamma_mults=args.l2_gamma_mults,
                alphas=args.alphas)

    print("Evaluating on test...")
    test_mmd_sqs = get_mmd_sq(
        test, train, gammas=l1_gamma, Y_diag=train_mmk_sqnorms[l1_gamma_i])
    K_test = make_kernel(test_mmd_sqs, l2_gamma, out=test_mmd_sqs)
    test_preds = model.predict(K_test) + bias

    test_mse = mean_squared_error(test.y, test_preds)
    print("Test MSE: {}".format(test_mse))
    test_r2 = r2_score(test.y, test_preds)
    print("Test R2: {}".format(test_r2))

    np.savez(
        os.path.join(args.out_dir, 'results.npz'),
        test_preds=test_preds, test_y=test.y,
        test_mse=test_mse, test_r2=test_r2,
        l1_gamma=l1_gamma, l2_gamma=l2_gamma, alpha=alpha,
        l1_gamma_mults=args.l1_gamma_mults, l1_gammas=l1_gammas,
        l2_gamma_mults=args.l2_gamma_mults, l2_gamma_scales=l2_gamma_scales,
        alphas=args.alphas, scores=scores,
        args=args)


def parse_args():
    def rest_of_args(subparser):
        params = subparser.add_argument_group("Hyperparameter grids")
        p = partial(params.add_argument, help="Default %(default)s.")
        p('--l1-gamma-mults', type=lambda s: np.asarray(eval(s)),
          default=default_l1_gamma_mults)
        p('--l2-gamma-mults', type=lambda s: np.asarray(eval(s)),
          default=default_l2_gamma_mults)
        p('--alphas', type=lambda s: np.asarray(eval(s)),
          default=default_alphas)

        io = subparser.add_argument_group("I/O parameters")
        i = io.add_argument
        i('out_dir')
        i('--cache-mmd', action='store_true', default=False)
        i('--no-cache-mmd', action='store_false', dest='cache-mmd')
        i('--n-jobs', type=int, default=-1,
          help="sklearn-style number of processes [%(default)s].")

    parser = make_parser(rest_of_args)
    args = parser.parse_args()
    if os.path.exists(args.out_dir):
        if set(os.listdir(args.out_dir)) - {'.DS_Store'}:
            parser.error(("Output directory {} exists, and I'm real scared of "
                          "overwriting data. Change the name or delete it.")
                         .format(args.out_dir))
    else:
        os.makedirs(args.out_dir)
    return args


def run_krr(train_sqdists, train_y, val_sqdists, val_y,
            l1_gammas, l2_gamma_mults=default_l2_gamma_mults,
            alphas=default_alphas):
    l2_gamma_scales = np.zeros_like(l1_gammas)

    n_l1_gammas = l1_gammas.size
    n_l2_gammas = l2_gamma_mults.size
    n_alphas = alphas.size
    n_train = train_sqdists.shape[1]
    n_val = val_sqdists.shape[1]
    assert train_sqdists.shape == (n_l1_gammas, n_train, n_train)
    assert val_sqdists.shape == (n_l1_gammas, n_val, n_train)

    scores = np.empty((n_l1_gammas, n_l2_gammas, n_alphas))
    scores.fill(np.nan)
    best_score = -np.inf
    best_param_is = None
    best_params = None
    best_model = None

    class CommaProgress(pb.widgets.WidgetBase):
        def __call__(self, progress, data):
            return '{value:,} of {max_value:,}'.format(**data)
    widgets = [' Tuning: ', CommaProgress(), ' (', pb.Percentage(), ') ',
               pb.Bar(), ' ', pb.ETA()]

    K = np.empty((n_train, n_train), dtype=np.float32)
    val_K = np.empty((n_val, n_train), dtype=np.float32)

    bar = pb.ProgressBar(maxval=n_l1_gammas * n_l2_gammas * n_alphas,
                         widgets=widgets)
    bar.start()
    for l1_gamma_i, (l1_gamma, train_D2, val_D2) in enumerate(
                zip(l1_gammas, train_sqdists, val_sqdists)):
        # get median of *these* squared distances, to scale l2 gamma
        med_sqdist = np.median(train_D2[np.triu_indices_from(train_D2, k=1)])
        l2_gamma_scales[l1_gamma_i] = med_sqdist

        for l2_gamma_i, l2_gamma in enumerate(l2_gamma_mults / med_sqdist):
            make_kernel(train_D2, l2_gamma, out=K)
            make_kernel(val_D2, l2_gamma, out=val_K)

            for alpha_i, alpha in enumerate(alphas):
                bar.update(alpha_i + l2_gamma_i * n_alphas
                           + l1_gamma_i * n_l2_gammas * n_alphas)

                model = KernelRidge(alpha=alpha, kernel='precomputed')
                model.fit(K, train_y)
                s = model.score(val_K, val_y)
                scores[l1_gamma_i, l2_gamma_i, alpha_i] = s

                if s > best_score:
                    best_param_is = (l1_gamma_i, l2_gamma_i, alpha_i)
                    best_params = (l1_gamma, l2_gamma, alpha)
                    best_score = s
                    best_model = model

    bar.finish()
    return best_model, best_params, best_param_is, scores, l2_gamma_scales


if __name__ == '__main__':
    main()
