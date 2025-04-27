#!/bin/bash
#$ -cwd
#$ -j y
#$ -l h_rt=1:0:0
#$ -l h_vmem=7.5G
#$ -l gpu=1
#$ -l gpu_type=ampere
#$ -N pretrain
#$ -m bea
#$ -l rocky
#$ -pe smp 8    
set -e

module load miniforge
mamba activate mae-cliport

python add_reverse_language_data.py --hdf5_path /data/home/acw694/CLIPort_new_loss/scratch/bridge_256_train.hdf5