# Python code for extract R results
import sys
import os 
import csv

import numpy as np
import glob

best_seed = []
for seed in range(23,33):
	validation = []
	name = []
	path = '/path/results-bdr-landmarks-conjugacy-l_*-s_*-k_*-g_true-d_astro-j_*-seed_{}.csv'.format(seed)
	for filename in glob.iglob(path):
		with open(filename, 'r') as file:
			reader = csv.reader(file, delimiter=',')
			results_list = list(reader)
			if len(results_list) == 0:
				continue
			else:
				validation.append(float(results_list[1][7]))
				name.append(filename)
	m = np.argmax(validation)
	best_seed.append(name[m])
	print('best file',name[m])
	print('best_Likelihood:', validation[m])

for i in range(0,10):
    os.system("cp " + best_seed[i] + " /data/ziz/hlaw/bdr/results_astro/" + method + "/seed_" + str(i + 23) +'.npz')