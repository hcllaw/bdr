# Python code for extract R results
import numpy as np
import glob
import sys
import os 
import csv

best_seed = []
for seed in range(23,33):
	validation = []
	name = []
	print('/data/ziz/hlaw/validate-results/results-bdr-landmarks-conjugacy-l_*-s_*-k_*-g_true-d_astro-j_*-seed_' +str(seed)+'.csv')
	for filename in glob.iglob('/data/ziz/hlaw/validate-results/results-bdr-landmarks-conjugacy-l_*-s_*-k_*-g_true-d_astro-j_*-seed_' +str(seed)+'.csv'):
		with open(filename, 'r') as my_file:
			print(filename)
			reader = csv.reader(my_file, delimiter=',')
			my_list = list(reader)
			if len(my_list) == 0:
				continue
			else:
				#print(my_list)
				validation.append(float(my_list[1][7]))
				name.append(filename)
	#print(filename)
	#print(validation)
	m = np.argmax(validation)
	best_seed.append(name[m])
	print('best',name[m])
	print('best_Likelihood:', validation[m])
	#print results_b.keys()

print(best_seed)
for i in range(0,10):
    os.system("cp " + best_seed[i] + " /data/ziz/hlaw/bdr/results_astro/" + method + "/seed_" + str(i + 23) +'.npz')