# Utilities for choosing best model based on validation set and moving them to the right location for plotting.
import sys
import os

import numpy as np
import glob
from scipy import stats

def results_mover(filelocation, target_dest, tune_using = 'nll'): # Tune based on NLL or MSE depending on usage
    validation = []
    name = []
    for result_file in glob.iglob(os.path.join(filelocation,'*', 'results.npz')):
        with np.load(result_file) as results:
            if tune_using == 'nll':
                if 'val_nll' in results:
                    validation.append(results['val_nll'])
                else:
                    raise('Val NLL not contained in results, please use MSE instead.')
            elif tune_using == 'mse':
                validation.append(results['val_mse'])
            else:
                raise('Please use either nll or mse')
        name.append(result_file)
    opt_index = np.argmin(validation)
    opt_name = name[opt_index]
    results = np.load(opt_name)
    print('test mse = %.03f'%results['test_mse'])
    if 'test_nll' in results:
        print('test nll = %.03f'%results['test_nll']) 
    os.system('cp {} {}'.format(opt_name, target_dest))
    print('Copied into {}'.format(target dest))

def main():
    files_dir = '/path/of/results/folder'
    target_dir = '/path/of/destination//vary_bag' #'/data/ziz/hlaw/bdr/results_chi2/big'
    target_folder = os.path.join(target_dir, 'radial') # change accordingly... 
    tune_method = 'nll'
    if not os.path.exists(target_folder):
        os.makedirs(target_folder) 

    for seed in range(23, 33):
        print('moving seed {}'.format(seed))
        filename = os.path.join(files_dir, 'radial_seed_{}_varybag'.format(seed))
        location_name = os.path.join(target_folder, 'seed_{}.npz'.format(seed))
        print('Filename = {} \n location_name = {}'.format(filename, location_name))
        results_mover(filename, location_name, tune_using = tune_method)
        print('----------------------------------')

if __name__ == '__main__':
    main()
