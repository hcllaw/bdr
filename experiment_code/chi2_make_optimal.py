# Generate the bayes optimal for the chi2 dataset script
#'Varying bag size: Uncertainty in the inputs.'
import os 
import subprocess

import numpy as np

seed = 23
for sbs in np.arange(0, 55, 5):
    out_dir = '/path/to/save/to/chi2_optimal_{}'.format(sbs)
    command = [
        "python", os.path.join("bayes_optimal.py"),
        'chi2',
        "--size-type","manual", #
        "--min-size", str(sbs),
        "--max-size", str(50 - sbs),
        "--data-seed", str(seed), 
        out_dir
    ]
    cmd = subprocess.list2cmdline(command)
    print(cmd)
    os.system(cmd)
