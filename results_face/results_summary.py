import os

from functools import partial
import numpy as np
import glob

def main():
    blr_nll = []
    blr_mse = []
    shrinkage_nll = []
    shrinkage_mse = []
    radial_mse = []
    bayes_nll = []
    bayes_mse = []
    avg_nll = []
    avg_mse = []
    files_dir = '/data/ziz/hlaw/bdr/results_face'
    for method in ['blr', 'shrinkage', 'radial', 'theirs_bayes', 'theirs_avg']:
        for seed in np.arange(20, 30, 1):
            target_file = os.path.join(files_dir, method, 'seed_{}.npz'.format(seed))
            with np.load(target_file) as results:
                if method == 'radial':
                    radial_mse.append(results['test_mse'])
                elif method == 'blr':
                    blr_mse.append(results['test_mse'])
                    blr_nll.append(float(results['test_nll']))
                elif method == 'shrinkage':
                    shrinkage_mse.append(results['test_mse'])
                    shrinkage_nll.append(float(results['test_nll']))
                elif method == 'theirs_bayes':
                    bayes_nll.append(float(results['test_nll']))
                    bayes_mse.append(results['test_mse'])
                elif method == 'theirs_avg':
                    avg_mse.append(results['test_mse'])
                    avg_nll.append(float(results['test_nll']))

    def summary(list_results, label='m'):
        if label == 'm':
            return np.mean(np.sqrt(list_results)), np.sqrt(np.var(np.sqrt(list_results)))
        elif label == 'n':
            return np.mean(list_results), np.sqrt(np.var(list_results))
    mse = partial(summary, label = 'm')
    nll = partial(summary, label = 'n')
    #print('Multiply Posterior RMSE', mse(bayes_mse))
    print('Baseline RMSE', mse(avg_mse))
    print('Radial RMSE:', mse(radial_mse))
    print('Blr RMSE:', mse(blr_mse ))
    print('Shrinkage RMSE:', mse(shrinkage_mse))
    print('-----------------')
    print('Blr NLL:', nll(blr_nll))
    print('Shrinkage NLL:', nll(shrinkage_nll))
    print('Baseline NLL:', nll(avg_nll))

    
if __name__ == '__main__':
    main()