import train_test
import os
import numpy as np
import sys

for std in np.arange(5,55,5):
	for dataset in ['chi2']: #'chi2','astro']:
	    print "Exporting %s into ../data/%s-varybagsize-%d/*"%(dataset,dataset,std)
	    sys.argv = ['export_for_stan.py', dataset, '--size-type', 'neg-binom', '--size-mean', '20', '--size-std', str(std), '--type', 'fourier', 'out']
	    if dataset == "chi2" or dataset == "chi2-varybagsize":
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

            os.system('mkdir ../data/%s-varybagsize-%d'%(dataset,std))
	    writecsv(train.features,train.y,'../data/%s-varybagsize-%d/train'%(dataset,std))
	    writecsv(test.features,test.y,'../data/%s-varybagsize-%d/test'%(dataset,std))
	    writecsv(val.features,val.y,'../data/%s-varybagsize-%d/val'%(dataset,std))
