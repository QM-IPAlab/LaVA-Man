#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=42
#SBATCH --mem-per-cpu=3850
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --account=su008-acw694

module load Miniconda3/4.12.0
source activate mae-cliport

python mae/dataset_co3d.py