import csv
from functools import partial
import os
import sys

import numpy as np

import train_test


for minsize in [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
    for dataset in ['chi2']:  # 'chi2','astro']:
        for size_name, size_args in [('', []),
                                     ('-small', ['--n-train', '150'])]:
            d = partial(
                os.path.join,
                '../data/{}-manual{}-{}'.format(dataset, size_name, minsize))
            print "Exporting {} into {}".format(dataset, d())
            sys.argv = ['export_for_stan.py', dataset,
                        '--size-type', 'manual', '--type', 'fourier', 'out',
                        '--min-size', str(minsize),
                        '--max-size', str(50-minsize),
                        '--data-seed', '47'] + size_args
            args = train_test.parse_args()
            train, _, val, test = train_test.generate_data(args)

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

            for n_landmarks in [30, 60]:
                args.n_landmarks = n_landmarks
                np.savetxt(d('landmarks-{}.csv'.format(n_landmarks)),
                           train_test.pick_landmarks(args, train))
