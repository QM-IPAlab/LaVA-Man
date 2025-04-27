#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3850
#SBATCH --time=08:00:00
#SBATCH --account=su008-acw694

module load Miniconda3/4.12.0
source activate mae-cliport

python unzip_target_files.py