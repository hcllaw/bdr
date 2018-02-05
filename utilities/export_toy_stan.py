# Export data for Stand for the BDR method... 
# Vary bag size and fixed bag size, need to export landmarks too
import csv
from functools import partial
import os
import sys

import numpy as np

import train_test

for seed in range(23, 33): # for minsize in np.arange(5,55,5):
    for dataset in ['chi2']: 
        d = partial(
            os.path.join,
            '/path/to/save/{}-seed-{}'.format(dataset, seed))
        print "Exporting {} into {}".format(dataset, d())
        sys.argv = ['export_for_stan.py', dataset,
                    '--size-type', 'special',  # change to manual for vary_bag_size
                    '--type', 'fourier', 'out', # any thing is ok 
                    #'--min-size', str(minsize),
                    #'--max-size', str(50-minsize),
                    #'--kmeans-landmarks',
                    '--data-seed', str(seed),
                    '--noise-std', str(1.0) # remove for vary_bag_size
                    ] + size_args
        args = train_test.parse_args()
        print(args)
        train, _, val, test = train_test.generate_data(args) # we dont use estop 

        def writecsv(x, y, fname):
            with open(fname + "_x.csv", "w") as f:
                fout = csv.writer(f, delimiter=",")
                for i in range(len(x)):
                    fout.writerows(np.c_[x[i], np.ones(len(x[i])) * i])
                f.close()
            np.savetxt(fname + '_y.csv', y, delimiter=",")

        if os.path.exists(d()):
            print("warning: {} already exists".format(d()))
        else:
            os.makedirs(d())
        writecsv(train.features, train.y, d('train'))
        writecsv(test.features, test.y, d('test'))
        writecsv(val.features, val.y, d('val'))

        for n_landmarks in [30, 60, 90]:
            args.n_landmarks = n_landmarks
            np.savetxt(d('landmarks-{}.csv'.format(n_landmarks)),
                       train_test.pick_landmarks(args, train))
