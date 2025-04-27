#!/bin/bash
#$ -pe smp 4
#$ -l h_vmem=8G
#$ -l h_rt=1:0:0
#$ -wd /data/home/acw694/CLIPort_new_loss
#$ -j y
#$ -N gen_hdf5
#$ -m bea
#$ -l rocky

set -e

# Replace the following line with a program or command
module load miniforge
mamba activate mae-cliport
python build_real_hdf5_multi_view.py -s train -o bridge_crossview_goal_3imgs_mask