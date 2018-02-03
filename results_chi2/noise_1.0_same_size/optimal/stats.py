#!/usr/bin/env python
from glob import glob
import numpy as np

mses = []
nlls = []

print("seed   % computed    MSE     NLL")
for x in sorted(glob('seed_*.npz')):
    seed = int(x[len('seed_'):-len('.npz')])
    with np.load(x) as d:
        mse = d['test_mse'][()]
        nll = d['test_nll'][()]
        pct_done = 1 if 'isnan' not in d else (1 - d['isnan'].mean())

        print("{:3}       {:4.0%}      {:.3f}   {:.3f}".format(seed, pct_done, mse, nll))

        mses.append(mse)
        nlls.append(nll)

print("MSE: {:.3f} ({:.3f})".format(np.mean(mses), np.std(mses)))
print("NLL: {:.3f} ({:.3f})".format(np.mean(nlls), np.std(nlls)))
