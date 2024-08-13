#!/bin/bash
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --job-name=gen_data
#SBATCH --cpus-per-task=16
module load python/3.8
source py-mae-cliport/bin/activate
export CLIPORT_ROOT=$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/mae

tasks=(
  'assembling-kits-seq-full'
  'packing-boxes-pairs-full'
  'put-block-in-bowl-full'
  'stack-block-pyramid-seq-full'
  'separating-piles-full'
  'towers-of-hanoi-seq-full'
  'packing-shapes'
  'align-rope'
)


for task in "${tasks[@]}"
do
    echo "Running data colloection for task: $task"
    python cliport/demos.py n=100 \
                            task=$task \
                            mode=val \
                            data_dir=data\
                            disp=False
done

# for task in "${tasks[@]}"
# do
#     echo "Running data colloection for task: $task"
#     python cliport/demos.py n=100 \
#                             task=$task \
#                             mode=test \
#                             data_dir=data\
#                             disp=False
# done