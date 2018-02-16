# Generate the bayes optimal for the chi2 dataset script
#'Varying bag size: Uncertainty in the inputs.'
#'Fixed bag size: Uncertainty in the regression model'
import os 

import subprocess
import numpy as np
# Change accordingly for fixed bag experiment, add the nosie 
seed = 23
for sbs in np.arange(0, 55, 5):
    out_dir = '/path/to/save/to/chi2_optimal_{}'.format(sbs)
    command = [
        "python", os.path.join("../experiment_code/bayes_optimal.py"),
        'chi2',
        "--size-type","manual", # change to special and add the noise option
        "--min-size", str(sbs),
        "--max-size", str(50 - sbs),
        "--data-seed", str(seed), 
        out_dir
    ]
    cmd = subprocess.list2cmdline(command)
    print(cmd)
    os.system(cmd)