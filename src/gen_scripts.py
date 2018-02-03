from os import system

header = """#!/bin/bash
#note anything after #SBATCH is a command
#SBATCH --mail-user=flaxman@gmail.com
#Email you if job starts, completed or failed
#SBATCH --mail-type=NONE
#SBATCH --job-name=%s-stan
#SBATCH --partition=large
#Choose your partition depending on your requirements
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=%d
#Memory per cpu in megabytes      
#SBATCH --output=/data/localhost/not-backed-up/flaxman/bdr_%%j.txt

set -x

"""

rscript = "Rscript bdr-landmarks.R -s %f -l %f -k %d -d %s -T true"

footer = """
echo "SLURM_JOBID: " $SLURM_JOBID

mv /data/localhost/not-backed-up/flaxman/*_${SLURM_JOBID}.txt /data/ziz/not-backed-up/flaxman/results/bdr/

"""

scripts = []
i = 0
for k in [100]:
	for sigma in [1e-2]:
	    for lengthscale in [.5, .75, 1.000000, 1.25]:
		for experiment in ["astro","chi2"]:
			fout = open("scripts/%d.slurm"%i,"w")
			
			ram = 50000
			fout.write(header%(experiment,ram))
			fout.write(rscript%(sigma,lengthscale,k,experiment))
			fout.write(footer)
			fout.close()
			scripts.append("sbatch scripts/%d.slurm"%i)
			i = i + 1

print i
import random
import time
##random.shuffle(scripts)
for i, s in enumerate(scripts):
    system(s)
    print i, s
