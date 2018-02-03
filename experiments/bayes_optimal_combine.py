import argparse
from glob import glob
import os
import sys

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

sys.path.append(os.path.dirname(__file__))
from train_test import generate_data


parser = argparse.ArgumentParser()
parser.add_argument('dir')
parser.add_argument('out', nargs='?')
parser.add_argument('--force', '-f', action='store_true', default=True)
args = parser.parse_args()

if args.out is None:
    args.out = (args.dir[:-1] if args.dir.endswith('/') else args.dir) + '.npz'
if not args.force:
    assert not os.path.exists(args.out)

d = {}
keys = ['preds', 'pred_vars', 'pred_nlls', 'pred_cdfs', 'pred_covered']
for k in keys:
    d['test_' + k] = v = np.empty(
            1000, dtype=bool if k == 'pred_covered' else float)
    v.fill(np.nan)

for fn in glob(os.path.join(args.dir, 'test_*.npz')):
    with np.load(fn, encoding='latin1') as inp:
        a = inp['args'][()]
        for k in ['skip_val', 'n_jobs', 'chunk_size', 'chunk_id']:
            if hasattr(a, k):
                delattr(a, k)
        if 'args' in d:
            if vars(a) != vars(d['args']):
                va = vars(a)
                vd = vars(d['args'])
                for k in set(va) | set(vd):
                    if va[k] != vd[k]:
                        print("differ in {}".format(k))
        else:
            d['args'] = a

        n = inp['ds_name'][()]
        if hasattr(n, 'decode'):
            n = n.decode()
        assert n == 'test'
        start, end = inp['ds_inds'][()]

        for k in keys:
            d['test_' + k][start:end] = inp[k]

isnan = np.isnan(d['test_preds'])
if isnan.any():
    print("ERROR: {} nan entries".format(isnan.sum()))
    print(np.isnan(d['test_preds']).nonzero()[0])
    d['isnan'] = isnan

_, _, _, test = generate_data(d['args'])
preds = d['test_preds']
d['test_y'] = test.y
d['test_mse'] = mean_squared_error(test.y[~isnan], preds[~isnan])
d['test_r2'] = r2_score(test.y[~isnan], preds[~isnan])
d['test_nll'] = d['test_pred_nlls'][~isnan].mean()
d['test_coverage'] = d['test_pred_covered'][~isnan].mean()

print("MSE: {}".format(d['test_mse']))
print("NLL: {}".format(d['test_nll']))

np.savez(args.out, **d)
