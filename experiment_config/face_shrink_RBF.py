# This file contains the configuration parameters for Face experiment for Shrinkage + RBF network

import os
from itertools import product
import subprocess

# Our parameters we will like to grid search over
seeds = range(20, 31)
structure = [64, 128, 256]
learning_rate = [0.001, 0.01, 0.05, 0.1, 0.5]
bandwidth_scales = [.5, .75, 1, 1.25, 1.5]
reg_para = [0, 0.000001] # early stopping so not really necessary

# Create a grid search vector using the itertools package
vector_grid = list(product(learning_rate, reg_para, structure, bandwidth_scales))
# take first one for example 
lr_c, reg_1_c, structure_c, bw_c = vector_grid[0]
batch_bags = 30
max_epochs = 750

for seed_c in seeds:
    folder_path = os.path.join(
            '/path/to/save/to',
            'shrinkage_face_seed_{}'.format(seed_c))
    command = [
                "python", "train_test.py",
                'imdb_faces',
                "-n", "shrinkage", # Change to radial for RBF network
                "--learning-rate", str(lr_c),
                "--n-landmarks", str(structure_c),
                "--no-opt-landmarks", 
                "--bw-scale", str(bw_c),
                "--reg-out", str(reg_1_c),
                "--max-epochs", str(max_epochs),
                "--batch-bags", str(batch_bags),
                '--shrink-towards-mean', #remove for RBF
                '--opt-R-scale', #remove for RBF
                '--no-cholesky', #remove for RBF
                '--empirical-cov-add-diag', str(0.01), #remove for RBF
                '--use-empirical-cov', #remove for RBF
                "--use-rbf-R", #remove for RBF
                "--rbf-R-bw-scale", str(1.0), #remove for RBF
                '--use-alpha-reg', #remove for RBF
                "--dtype-double",
                "--split-seed", str(seed_c),
                folder_path
                ]
    cmd = subprocess.list2cmdline(command)
    print(cmd)
    os.system(cmd)
