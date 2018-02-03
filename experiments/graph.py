from __future__ import division, print_function
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats

sns.set_style('whitegrid')
sns.set_context('paper', font_scale=1.5)

import sys
sys.path.append('../experiments/')
sys.path.append('/data/ziz/hlaw/bdr/results_chi2')

from train_test import make_parser, generate_data


#bdr_mse = [0.178831195720528, 0.238735696296155, 0.281660477372944, 0.313499823879328,
# 0.368847120960408, 0.430017663313621, 0.478589870258002, 0.557061947733941, 
# 0.532174569425381, 0.613619291871497, 0.688052526198335]

#bdr_nll =  [0.211272791853874, 0.295317312649724, 0.417896011425974,
#                   0.484746674046461, 0.582620914966886 ,0.707657784022924,
#                   0.785941626682198, 0.89156270209381, 0.921763794484579,
#                   1.04954872155933, 1.14834811852655]

_data = {}

def get_data(sz, prop_min):
    if (sz, prop_min) in _data:
        return _data[sz, prop_min]
    
    parser = make_parser()
    args = ['chi2', '--type=fourier', 'out']
    if sz == 'big':
        args += ['--data-seed=23']
    elif sz == 'small':
        args += ['--data-seed=23', '--n-train=150']
    else:
        raise ValueError()
    args += ['--min-size', str(prop_min), '--max-size', str(50 - prop_min),
             '--size-type', 'manual']
    
    args = parser.parse_args(args)
    train, estop, val, test = _data[sz, prop_min] = generate_data(args)
    return train, estop, val, test

from glob import glob
import os
import re

data = []

pat = re.compile('(big|small)/(.*)/(.*)\.npz')
for fn in glob('*/*/*.npz'):
    print(fn)
    sz, method, prop_small = pat.match(fn).groups()
    prop_small = int(prop_small)
    
    train, estop, val, test = get_data(sz, prop_small)
    
    with np.load(fn) as d:
        assert np.allclose(test.y, d['test_y'])
        means = d['test_preds']
        if method == 'optimal_23':
            stds = np.sqrt(d['test_pred_vars'])
            nll = d['test_pred_nlls']
        elif 'test_preds_var' in d:
            stds = np.sqrt(d['test_preds_var'])
            nll = -stats.norm.logpdf(test.y, means, stds)
        else:
            stds = nll = np.nan
        data.append(pd.DataFrame.from_dict(dict(
            ds_size=sz,
            prop_small=prop_small,
            method=method,
            pt_idx=np.arange(len(test)),
            y=test.y,
            pred=means,
            pred_std=stds, 
            pred_nll=nll,   # Check THIS 
            #pred_nll=d['test_nll'],
            n_pts=test.n_pts,
        )))

#pat = re.compile('(big|small)/(.*)/.*-(\d+)-j_\d+\.csv')
pat = re.compile('(big|small)/(.*)/.*chi2-manual-(\d+)-i.*\.csv')
for fn in glob('*/*/results-*.csv'):
    print(fn)
    sz, method, prop_small = pat.match(fn).groups()
    prop_small = int(prop_small)
    train, estop, val, test = get_data(sz, prop_small)
    
    d = pd.read_csv(fn)
    assert len(d) == len(train) + len(val) + len(test)
    d = d[-len(test):]
    d.rename(columns={'mu': 'pred', 'sd': 'pred_std', 'lp': 'pred_nll'}, inplace=True)
    d.pred_nll *= -1
    d['ds_size'] = sz
    d['prop_small'] = prop_small
    d['method'] = method
    d['pt_idx'] = np.arange(len(test))
    d['n_pts'] = test.n_pts
    data.append(d)
        
data = pd.concat(data).set_index(['ds_size', 'prop_small', 'method', 'pt_idx']).sort_index()

data['sqerr'] = (data.y - data.pred) ** 2

data.xs(0, level='pt_idx').xs(0, level='prop_small')

for sz in ['big']:#,'small']:
    plt.figure()
    props = data.index.get_level_values('prop_small').unique()
    plt.plot(props, [get_data(sz, p)[-1].y.var() for p in props], color='k', ls='--', lw=1,
             label='mean label')
    #plt.plot(np.arange(0,55,5), bdr_nll, color = 'y', marker= 'o', label = 'BDR')
    g = lambda m: data.sqerr.loc[sz].xs(m, level='method').groupby(level=0).mean()
    g('optimal_23').plot(color='k', label='optimal')   
    g('bdr').plot(color='y', marker= 'o', label = 'BDR')     
    g('radialblr').plot(color='c', marker='o', label='BLR')
    g('shrinkage_rbf_1.0').plot(color='g', marker='s', label='shrinkage_rbf')
    g('radial').plot(color='m', marker='s', label='radial')
    if sz == 'big':
        g('shrinkage_rbf_real_r').plot(color='r', label='shrinkage_real_r')
    plt.legend(loc=(.02, .52))
    plt.ylabel('MSE')
    plt.xlabel('proportion with $n=5$')
    from matplotlib.ticker import FuncFormatter
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, pos: '{:.0%}'.format(x / 100.)))
    plt.show()

for sz in ['big']:
    plt.figure()
    props = data.index.get_level_values('prop_small').unique()
    unifs = np.empty(props.shape)
    for i, p in enumerate(props):
        _, _, _, test = get_data(sz, p)
        unifs[i] = np.log(test.y.max() - test.y.min())
    plt.plot(props, unifs, color='k', ls='--', lw=1, label='uniform')
    plt.plot(np.arange(0,55,5), bdr_nll, color = 'y', marker= 'o', label = 'BDR')
    g = lambda m: data.pred_nll.loc[sz].xs(m, level='method').groupby(level=0).mean()
    g('optimal_23').plot(color='k', label='optimal')
    g('radialblr').plot(color='c', marker='o', label='BLR')
    g('bdr').plot(color='y', marker= 'o', label = 'BDR')     
    g('shrinkage_rbf_1.0').plot(color='g', marker='s', label='shrinkage_rbf')
    if sz == 'big':
        g('shrinkage_rbf_real_r').plot(color='r', label='shrinkage_real_r')
    plt.legend(loc='best')
    plt.ylabel('predictive mean NLL')
    plt.xlabel('proportion with $n=5$')
    from matplotlib.ticker import FuncFormatter
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, pos: '{:.0%}'.format(x / 100.)))
    plt.show()
    #plt.savefig('../drafts/figs/nll_{}.pdf'.format(sz), bbox_inches='tight', pad_inches=0)

sizes = np.r_[np.nan, np.sort(data.n_pts.unique())]

method_names = np.array(['BLR', 'shrinkage', 'BDR', 'optimal'])
methods = np.array(['radialblr', 'shrinkage_rbf_1.0', 'bdr', 'optimal_23'])

sub = data.loc['big', 25].loc[list(methods)]

plt.figure(figsize=(7.9, 9.5), dpi=200)

for i, s in enumerate(sizes):
    sub_size = sub if np.isnan(s) else sub[sub.n_pts == s]
    
    std_max = sub_size.pred_std.max()
    ylo = sub_size.pred.min()
    yhi = sub_size.pred.max()
    
    for j, (m, m_name) in enumerate(zip(methods, method_names)):
        bit = sub_size.loc[m]
        plt.subplot(sizes.size, methods.size, i * methods.size + j + 1)
        
        if i == 0:
            plt.title(m_name)
        if j == 0:
            plt.ylabel(r'$N_j = ${}'.format('all' if i == 0 else int(s)))
            
        plt.plot([4, 8], [4, 8], color='k')
        plt.scatter(bit.y, bit.pred, c=bit.pred_std, cmap='viridis', vmin=0, vmax=std_max,
                   alpha=.7)
        
        plt.grid(False)
        if i != sizes.size - 1:
            plt.xticks([], [])
        if j != 0:
            plt.yticks([], [])
        if j == methods.size - 1:
            plt.colorbar()
        
        plt.xlim(4, 8)
        plt.ylim(ylo, yhi)
        
        plt.annotate('MSE = {: .3f}'.format(bit.sqerr.mean()),
                     xy=(.98, 0.02), xycoords='axes fraction', ha='right', va='bottom',
                     fontsize=10)
        plt.annotate('NLL = {: .3f}'.format(bit.pred_nll.mean()),
                     xy=(.98, 0.12), xycoords='axes fraction', ha='right', va='bottom',
                     fontsize=10)
plt.tight_layout(pad=0)
plt.savefig('../drafts/figs/preds.pdf', bbox_inches='tight', pad_inches=0)