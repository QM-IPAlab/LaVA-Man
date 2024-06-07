##!/bin/bash
##SBATCH --partition=small
##SBATCH --gres=gpu:1
##SBATCH --job-name=rlang
##SBATCH --cpus-per-task=16
#module load python/anaconda3
#source activate mae-cliport
export CLIPORT_ROOT=$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/mae




python cliport/demos.py n=10 \
                        task=stack-block-pyramid-seq-seen-colors \
                        mode=test