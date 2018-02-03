from os import system

header = """#!/bin/bash
#note anything after #SBATCH is a command
#SBATCH --mail-user=flaxman@gmail.com
#Email you if job starts, completed or failed
#SBATCH --mail-type=ALL
#SBATCH --job-name=%s-%s
#SBATCH --partition=small
#Choose your partition depending on your requirements
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=%d
#Memory per cpu in megabytes      
#SBATCH --output=/data/localhost/not-backed-up/flaxman/bdr_%%j.txt

set -x

"""

rscript = "Rscript bdr-landmarks-map.R -s %f -l %f -k %d -d %s -m %s -g %s -S true"

footer = """
echo "SLURM_JOBID: " $SLURM_JOBID

mv /data/localhost/not-backed-up/flaxman/*_${SLURM_JOBID}.txt /data/ziz/not-backed-up/flaxman/results/bdr/

"""

scripts = []
i = 0
for g in ['true']: # global mean?
    for model in ['bdr-landmarks-blr']: #,'bdr-landmarks-blr','bdr-landmarks-conjugacy']: #,'bdr-landmarks-shrinkage']:
		for k in [60]:
			for sigma in [1e-2]:
			    for lengthscale in [1]: #[.5, .75, 1.000000, 1.25]:
                                for experiment in ["chi2-manual-small-5","chi2-manual-small-0","chi2-manual-small-10","chi2-manual-small-15","chi2-manual-small-20","chi2-manual-small-30","chi2-manual-small-40","chi2-manual-small-25","chi2-manual-small-35","chi2-manual-small-45","chi2-manual-small-50"]: #,"chi2-manual-small-40","chi2-manual-small-45","chi2-manual-small-50"]: #"chi2","astro"]:
					fout = open("scripts/%d.slurm"%i,"w")
					
					ram = 20000
					fout.write(header%(model,experiment,ram))
					fout.write(rscript%(sigma,lengthscale,k,experiment,model,g))
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
