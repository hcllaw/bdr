import sys
import os

import numpy as np
import glob
from scipy import stats

def results_mover(filename, location_name, tune_using = 'nll'):
    validation = []
    name = []
    for result_file in glob.iglob(os.path.join(filename,'*', 'results.npz')):
        with np.load(result_file) as results:
            if tune_using == 'nll':
                validation.append(results['val_mse'])
            elif tune_using == 'mse':
                validation.append(results['val_nll'])
            else:
                raise('Please use either nll or mse')
        name.append(result_file)
    opt_index = np.argmin(validation)
    opt_name = name[opt_index]
    results = np.load(opt_name)
    print('test mse = %.03f'%results['test_mse'])    
    print('test nll = %.03f'%results['test_nll'])
    os.system('cp {} {}'.format(opt_name, location_name))
    print('Moved to {}'.format(location_name))

def main():
    files_dir = '/data/ziz/hlaw/results_aistats'
    target_dir = '/data/ziz/hlaw/bdr/results_chi2/small'
    target_folder = os.path.join(target_dir, 'optimal_23')
    
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for sbs in np.arange(0,55,5):
        print('moving bag size {}'.format(sbs))
        filename = os.path.join(files_dir, 'chi2_optimal_small_{}'.format(sbs))
        location_name = os.path.join(target_folder, '{}.npz'.format(sbs))
        print('Filename = {} \n location_name = {}'.format(filename, location_name))
        results_mover(filename, location_name)
        print('----------------------------------')

if __name__ == '__main__':
    main()
