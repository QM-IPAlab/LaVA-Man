#!/bin/bash
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --job-name=gen_data
#SBATCH --cpus-per-task=16
#SBATCH --array=0-1
#SBATCH --time=24:00:00
module load python/anaconda3
source activate mae-cliport
export CLIPORT_ROOT=$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/mae

tasks=(
  'separating-piles-unseen-colors'
  'separating-piles-seen-colors'
)

task_name=${tasks[$SLURM_ARRAY_TASK_ID]}

python cliport/demos.py n=1000 \
                        task=${task_name} \
                        mode=train
