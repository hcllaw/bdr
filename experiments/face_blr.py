# This file contains the configuration parameters for Face experiment for BLR

import os
from itertools import product
import subprocess

# Our parameters we will like to grid search over
seeds = range(20, 31)
structure = [64, 128, 256]
learning_rate = [0.01, 0.05, 0.1, 0.5, 1.0] 
bandwidth_scales = [.5, .75, 1, 1.25, 1.5]
batch_bags = 30
max_epochs = 750
# Create a grid search vector using the itertools package
vector_grid = list(product(structure, learning_rate, bandwidth_scales))

# take first one for example 
structure_c, lr_c, bw_c = vector_grid[0]

for seed in seeds:
    folder_path = os.path.join(
        '/path/to/save/to',
            'blr_face_seed_{}'.format(seed))
    # Save output and parameters to text file in the localhost node,
    # which is where the computation is performed.
    command = [
        "python", os.path.join(dirname, "blr.py"),
        'imdb_faces',
        "--type", "radial-precomp",
        "--learning-rate", str(lr_c),
        "--n-landmarks", str(structure_c),
        "--no-opt-landmarks", "--bw-scale", str(bw_c),
        "--max-epochs", str(max_epochs),
        "--dtype-double",
        "--split-seed", str(seed),
        folder_path
            ]
    cmd = subprocess.list2cmdline(command)
    print(cmd)
    os.system(cmd)