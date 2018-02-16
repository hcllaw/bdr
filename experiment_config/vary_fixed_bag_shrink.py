# This file contains the configuration parameters for shrinkageC, shrinkage
#'Varying bag size: Uncertainty in the inputs.'
#'Fixed bag size: Uncertainty in the regression model'

import os
from itertools import product
import subprocess

# Our parameters we will like to grid search over
seeds = range(23, 33) # loop over if fixed bag size instead of sbs
sbs = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
structure = [30, 60, 90]
learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.5]
reg_1 = [0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
bandwidth_scales = [.5, .75, 1, 1.25, 1.5] 
batch_bags = 30
seed = 23
max_epochs = 750

# Create a list of combinations 
vector_grid = list(product(structure, learning_rate, reg_1, bandwidth_scales))

# let structure_c, lr_c, reg_1_c, bw_c be the particular chosen configurations, in this case take first one
structure_c, lr_c, reg_1_c, bw_c = vector_grid[0]

# Over different sizes for the bags, loop over seeds instead for fixed bag size
# and turn on appropriate optionsfor sbs_size in sbs:
for sbs_size in sbs:
    folder_path = os.path.join(
        '/path/to/save/to',
        'shrink_chi2_size_{}'.format(sbs_size))
    command = [
        'python', '../experiment_code/train_test.py',
        'chi2', # dataset 
        '-n', 'shrinkage',
        '--size-type','manual',
        #"--size-type","special", # turn on if fixed bag size
        #'--noise-std', str(1.0), # turn on if fixed bag size
        '--min-size', str(sbs_size), # turn off if fixed bag size
        '--max-size', str(50 - sbs_size),# turn off if fixed bag size
        '--learning-rate', str(lr_c),
        '--n-landmarks', str(structure_c),
        '--no-opt-landmarks', 
        '--bw-scale', str(bw_c),
        '--reg-out', str(reg_1_c),
        '--max-epochs', str(max_epochs),
        '--batch-bags', str(batch_bags),
        '--shrink-towards-mean', # m_0 to shrink to 
        '--opt-R-scale', # strength of shrinkage
        '--kmeans-landmarks', # landmarks - use k means to choose, turn off for fix bag size
        '--use-empirical-cov', #avg of empirical covariance of bag
        '--empirical-cov-add-diag', str(0.001), #stability
        '--use-rbf-R', #for shrinkage 
        '--rbf-R-bw-scale', str(1.0), #for shrinkage
        #'--use-real-R', ShrinkageC
        '--use-alpha-reg', # RKHS reg
        '--dtype-double',
        '--data-seed', str(seed),
        folder_path
    ]
    cmd = subprocess.list2cmdline(command)
    print(cmd)
    os.system(cmd)