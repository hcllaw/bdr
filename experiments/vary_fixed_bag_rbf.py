# This file contains the configuration parameters for RBF network
#'Varying bag size: Uncertainty in the inputs'
#'Fixed bag size: Uncertainty in the regression model'
import os
from itertools import product
import subprocess

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
# and turn on appropriate options
for sbs_size in sbs:
    folder_path = os.path.join(
        '/path/to/save/to',
        'rbf_chi2_size_{}'.format(sbs_size))
    command = [
        'python train_test.py',
        'chi_2', # dataset 
        '-n', 'radial',
        '--size-type','manual', 
        #"--size-type","special", # turn on if fixed bag size
        #'--noise-std', str(1.0), # turn on if fixed bag size
        "--min-size", str(sbs_size), # turn off if fixed bag size
        "--max-size", str(50 - sbs_size), # turn off if fixed bag size
        '--learning-rate', str(lr_c),
        '--n-landmarks', str(structure_c),
        '--no-opt-landmarks', 
        '--bw-scale', str(bw_c),
        '--reg-out', str(reg_1_c),
        '--max-epochs', str(max_epochs),
        '--batch-bags', str(batch_bags),
        '--kmeans-landmarks', # turn off if fixed bag size
        '--dtype-double',
        '--data-seed', str(seed),
        folder_path
    ]
    cmd = subprocess.list2cmdline(command)
    print(cmd)
    os.system(cmd)
