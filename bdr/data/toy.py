from __future__ import division

import numpy as np
from scipy import stats
from sklearn.utils import check_random_state

from bdr import Features


def generate_chisq(num_bags, dim, bag_sizes=[50, 100], df_range=[4, 8],
                   noise=0.0, size_type='uniform',
                   manual_num=[5, 20, 100, 1000], seed=None):
    rs = check_random_state(seed)
    
    if size_type == 'uniform':
        lo, hi = bag_sizes
        sizes = rs.randint(low=lo, high=hi + 1, size=num_bags)
    elif size_type == 'neg-binom':
        # Do a negative binomial + 1 (in Wikipedia's notation),
        # so that sizes are a distribution on the positive integers.
        # mean = p r / (1 - p) + 1; var = (mean - 1) / (1 - p)
        mean, std = bag_sizes
        p = 1 - (mean - 1) / (std * std)
        assert 0 < p < 1
        r = (mean - 1) * (1 - p) / p
        assert r > 0
        # scipy swaps p and 1-p
        sizes = stats.nbinom(r, 1 - p).rvs(size=num_bags) + 1
    elif size_type == 'manual':
        m1, m2, m3, m4 = manual_num
        s1, s2, s3, s4 = bag_sizes
        assert np.allclose(s1 + s2 + s3 + s4, 1)
        sizes = np.r_[
            np.repeat(m1, int(num_bags * s1)),
            np.repeat(m2, int(num_bags * s2)),
            np.repeat(m3, int(num_bags * s3)),
        ]
        sizes = np.r_[sizes, np.repeat(m4, num_bags - sizes.size)]
        rs.shuffle(sizes)
    else:
        raise ValueError("unknown size_type {}".format(size_type))
    y = rs.uniform(low=df_range[0], high=df_range[1], size=num_bags)

    def make(n, df):
        return (rs.chisquare(df=df, size=(n, dim)) / df
                + rs.normal(scale=np.sqrt(noise), size=(n, dim)))
    return Features([make(n, df) for n, df in zip(sizes, y)], y=y)
