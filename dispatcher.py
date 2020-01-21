'''This is a function to run many different instances on the cluster
'''

# scp -r ~/Documents/GeorgTravisRot/POE_learner jjd65@erisone.partners.org:/lsf
import numpy as np
import os
import shutil
import argparse

measurement_noises = [0.01, 0.05, 0.1, 0.15]
use_mm = [0,1]


data_seed = np.arange(0,args.n_data_seeds)
init_seed = np.arange(0,args.n_init_seeds)
if args.basepath is None:
    basepath = 'OTUs{}/'.format(args.n_otus)
else:
    basepath = args.basepath

my_str = '''
#!/bin/bash
#BSUB -J pylab
#BSUB -o {3}.out
#BSUB -e {3}.err

# This is a sample script with specific resource requirements for the
# **bigmemory** queue with 64GB memory requirement and memory
# limit settings, which are both needed for reservations of
# more than 40GB.
# Copy this script and then submit job as follows:
# ---
# cd ~/lsf
# cp templates/bsub/example_8CPU_bigmulti_64GB.lsf .
# bsub < example_bigmulti_8CPU_64GB.lsf
# ---
# Then look in the ~/lsf/output folder for the script log
# that matches the job ID number

# Please make a copy of this script for your own modifications

#BSUB -q big-multi
#BSUB -n 8
#BSUB -M 10000
#BSUB -R rusage[mem=10000]

# Some important variables to check (Can be removed later)
echo '---PROCESS RESOURCE LIMITS---'
ulimit -a
echo '---SHARED LIBRARY PATH---'
echo $LD_LIBRARY_PATH
echo '---APPLICATION SEARCH PATH:---'
echo $PATH
echo '---LSF Parameters:---'
printenv | grep '^LSF'
echo '---LSB Parameters:---'
printenv | grep '^LSB'
echo '---LOADED MODULES:---'
module list
echo '---SHELL:---'
echo $SHELL
echo '---HOSTNAME:---'
hostname
echo '---GROUP MEMBERSHIP (files are created in the first group listed):---'
groups
echo '---DEFAULT FILE PERMISSIONS (UMASK):---'
umask
echo '---CURRENT WORKING DIRECTORY:---'
pwd
echo '---DISK SPACE QUOTA---'
df .
echo '---TEMPORARY SCRATCH FOLDER ($TMPDIR):---'
echo $TMPDIR

# Add your job command here
# Load module
module load anaconda/default
source activate dispatcher

cd /PHShome/dek15/perturbation_study/
python3 ./main.py -MM {0} -a 1 -b 0.5 -nb 3 -mv {1} -pv 1 -tv 1 -bv 1 -av 1 -poe 1 -no 2 -ns 500 -dt 0.01 -o {2}
'''

# Make the directories to store the information

#try:
#    os.mkdir(basepath)
#except OSError:
    # Remove the directory and then make one
#    shutil.rmtree(basepath)
#    os.mkdir(basepath)
basepath = 'outdir'

for m in use_mm:

    for mn in measurement_noises:
        outdir = 'outdir_mvar' + str(mn).replace('.', ',') + '_MM' + str(m)

        fname = outdir + '.lsf'

        f = open(fname,'w')
        f.write(my_str.format(m,mn,outdir,outdir))
        f.close()

        mnfolder = basepath + 'measurement_noise_{}/'.format(int(100*mn))
        os.mkdir(mnfolder)

        os.system('bsub < {}'.format(fname))
