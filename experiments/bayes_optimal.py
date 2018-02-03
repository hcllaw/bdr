#!/usr/bin/env python
from __future__ import division, print_function
import os
import sys

import numpy as np
from scipy.integrate import quad
from scipy.misc import logsumexp
from scipy.special import gammaln
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals.joblib import Parallel, delayed
from tqdm import tqdm

sys.path.append(os.path.dirname(__file__))
from train_test import make_parser, generate_data, check_output_dir


def log_likelihood(X, k):
    return X.size * (
        np.log(k)
        - k/2 * np.log(2)
        - gammaln(k / 2)
        + (k/2 - 1) * (np.log(k) + np.log(X).mean())
        - k/2 * np.mean(X)
    )

_integrand = None
def make_integrand():
    global _integrand
    import numba
    import numba.types as nt
    import scipy

    @numba.cfunc(nt.double(nt.intc, nt.CPointer(nt.double)),
                 nopython=True, nogil=True)
    def _fn(n, xx):
        if n != 4:
            return np.nan
        y = xx[0]
        k = xx[1]
        sigma2 = xx[2]
        z = xx[3]
        return y ** (k/2 - 1) * np.exp(-(k * y + (z-y)**2 / sigma2) / 2)
    _integrand = scipy.LowLevelCallable(_fn.ctypes)

    #_integrand = lambda y, k, sigma2, z: y ** (k/2 - 1) * np.exp(-(k * y + (z-y)**2 / sigma2)/2)

def just_quad(*a, **kw):
    return quad(*a, **kw)[0]

def log_likelihood_noise(X, k, sigma, n_jobs=1):
    if _integrand is None:
        make_integrand()
    
    X = np.ravel(X)

    scalar_k = np.isscalar(k)
    k = np.ravel([k])[None, :, None].astype(X.dtype)
    assert np.all(k > 0)

    scalar_sigma = np.isscalar(sigma)
    sigma2 = np.ravel([sigma])[None, None, :].astype(X.dtype) ** 2
    del sigma
    assert np.all(sigma2 > 0)

    const = X.size * (
        k/2 * np.log(k/2)
        - gammaln(k/2)
        - np.log(2 * np.pi * sigma2) / 2
    )
    assert const.ndim == 3
    assert const.shape[0] == 1
    const = const[0]

    res = np.empty((k.size, sigma2.size))
    res.fill(np.nan)

    with Parallel(n_jobs=n_jobs) as par:
        for ki, kval in enumerate(k.ravel()):
            for si, sval in enumerate(sigma2.ravel()):
                bits = par(
                    delayed(just_quad)(_integrand, 0, np.inf, args=(kval, sval, x))
                    for x in X)
                res[ki, si] = const[ki, si] + np.log(bits).sum()

    if scalar_k:
        res = res[0]
    if scalar_sigma:
        res = res[..., 0]
    return res


def interval(p, dfs, center, coverage=.95):
    # assumes that p sums to 1, instead of integrates to 1
    i = dfs.searchsorted(center)
    # interval: covers [max(i - w, 0), min(i + w, len(p))]
    # want smallest w such that interval sum is >= coverage

    cumsum = np.r_[0, np.cumsum(p)]
    # sum(interval) == cumsum[min(i+w+1, len(p))] - cumsum[max(i-w, 0)]
    # find all sums:
    uppers = np.r_[cumsum[i+1:], np.repeat(cumsum[-1], i)]
    lowers = np.r_[cumsum[i:0:-1], np.repeat(cumsum[0], len(p) - i)]
    w = np.searchsorted(uppers - lowers, coverage)

    return dfs[max(i - w, 0)], dfs[min(i + w + 1, len(p)) - 1]


def posterior_info(X, dfs, y, sigma=0, n_jobs=1):
    if sigma == 0:
        ll = log_likelihood(X, dfs)
    else:
        ll = log_likelihood_noise(X, dfs, sigma, n_jobs=n_jobs)

    # "posterior" that sums to 1 is better for most of what we want
    log_posterior = ll - logsumexp(ll)
    posterior = np.exp(log_posterior)

    # the actual posterior needs to integrate to 1, though.
    # this will be off by a factor of np.diff(dfs).mean(),
    # assuming that dfs have constant spacing

    mean = posterior.dot(dfs)
    assert np.isfinite(mean)
    assert dfs[0] <= mean <= dfs[-1]

    var = posterior.dot(dfs ** 2) - mean**2

    y_pos = dfs.searchsorted(y)

    nll = -(log_posterior[y_pos] - np.log(np.diff(dfs).mean()))

    lo, hi = interval(posterior, dfs, mean)

    cdf = np.exp(logsumexp(log_posterior[:y_pos]))

    return mean, var, nll, lo < y < hi, cdf


def get_posterior_preds(feats, y, dfs, args):
    res = Parallel(n_jobs=args.n_jobs, verbose=args.verbosity)(
            delayed(posterior_info)(bag, dfs, y, args.noise_std)
            for bag, y in zip(feats, feats.y))
    return tuple(np.array(x) for x in zip(*res))


def _rest_of_args(subparser):
    subparser.add_argument('out_dir')
    subparser.add_argument('chunk_id', type=int, nargs='?')
    subparser.add_argument('--chunk-size', type=int, default=10)
    subparser.add_argument('--n-discretization', type=int, default=5000)
    subparser.add_argument('--n-jobs', type=int, default=1)
    subparser.add_argument('--skip-val', action='store_true', default=False)
    subparser.add_argument('--verbosity', type=int, default=4)


def main():
    parser = make_parser(_rest_of_args)
    args = parser.parse_args()
    if args.dataset != 'chi2':
        parser.error("Only know bayes-optimal answer for chi2.")


    _, _, val, test = generate_data(args)
    dfs = np.linspace(args.min_df, args.max_df, args.n_discretization)

    d = {'args': args}

    if args.chunk_id is not None:
        i = args.chunk_id
        n = args.chunk_size
        n_val_jobs = int(np.ceil(len(val) / n))
        if i < n_val_jobs:
            start = i * n
            end = min((i + 1) * n, len(val))
            assert end > start
            ds = val[start:end]
            d['ds_name'] = name = 'val'
            d['ds_inds'] = (start, end)
        else:
            i -= n_val_jobs
            start = i * n
            end = min((i + 1) * n, len(test))
            assert end > start
            ds = test[start:end]
            d['ds_name'] = name = 'test'
            d['ds_inds'] = (start, end)

        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)
        fn = os.path.join(args.out_dir, '{}_{:04}_{:04}.npz'.format(name, start, end))
        assert not os.path.exists(fn)

        (d['preds'], d['pred_vars'], d['pred_nlls'], d['pred_covered'],
         d['pred_cdfs']) = get_posterior_preds(ds, ds.y, dfs, args)

        np.savez(fn, **d)

    else:
        check_output_dir(args.out_dir, parser)
        for name, ds in [('val', val), ('test', test)]:
            if args.skip_val and name == 'val':
                continue
            print("Starting {}: {}".format(name, ds))
            preds, vars, pred_nlls, coverage, pred_cdfs = \
                    get_posterior_preds(ds, ds.y, dfs, args)
            d['{}_y'.format(name)] = ds.y
            d['{}_preds'.format(name)] = preds
            d['{}_pred_vars'.format(name)] = vars
            d['{}_pred_nlls'.format(name)] = pred_nlls
            d['{}_pred_cdfs'.format(name)] = pred_cdfs
            d['{}_mse'.format(name)] = mean_squared_error(ds.y, preds)
            d['{}_r2'.format(name)] = r2_score(ds.y, preds)
            d['{}_nll'.format(name)] = pred_nlls.mean()
            d['{}_coverage'.format(name)] = np.mean(coverage)

        np.savez(os.path.join(args.out_dir, 'results.npz'), **d)


if __name__ == '__main__':
    main()
