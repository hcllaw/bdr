import train_test
import sys

for dataset in ['chi2','astro']:
    print "Exporting %s into ../data/%s/*"%(dataset,dataset)
    sys.argv = ['export_for_stan.py', dataset, '--type', 'fourier', 'out']
    if dataset == "chi2":
        sys.argv += ['--data-seed', '23']
    else:
        sys.argv += ['--split-seed', '23']
    args = train_test.parse_args()
    train, _, val, test = train_test.generate_data(args)

    import csv
    import numpy as np
    def writecsv(x,y,fname):
        f = open(fname + "_x.csv","w")
        fout = csv.writer(f,delimiter=",")
        for i in range(len(x)):
            fout.writerows(np.c_[ x[i], np.ones(len(x[i])) * i ])
        f.close()
        np.savetxt(fname + '_y.csv', y, delimiter=",")

    writecsv(train.features,train.y,'../data/%s/train'%dataset)
    writecsv(test.features,test.y,'../data/%s/test'%dataset)
    writecsv(val.features,val.y,'../data/%s/val'%dataset)
