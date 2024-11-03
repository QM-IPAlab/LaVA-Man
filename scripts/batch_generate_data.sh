#!/bin/bash
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --job-name=gen_data
#SBATCH --cpus-per-task=32

module load python/3.8
source py-mae-cliport/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/mae
export CLIPORT_ROOT=$(pwd)

# python -m cliport.demos_top_down n=1 \
#                         task=packing-unseen-google-objects-group \
#                         mode=train\
#                         save_type=hdf5

# declare -a tasks=("put-block-in-bowl-seen-colors"\
#     "put-block-in-bowl-unseen-colors"\
#     "packing-seen-google-objects-group"\
#     "packing-unseen-google-objects-group"\
#     "packing-seen-google-objects-seq"\
#     "packing-unseen-google-objects-seq"\
# )


# tasks=(
#   'assembling-kits-seq-full'
#   'packing-boxes-pairs-full'
#   'put-block-in-bowl-full'
#   'stack-block-pyramid-seq-full'
#   'separating-piles-full'
#   'towers-of-hanoi-seq-full'
#   'packing-shapes'
#   'align-rope'
# )

tasks=(
  'align-rope'
  'assembling-kits-seq-full'
  'packing-boxes-pairs-full'
  'packing-shapes'
  'packing-unseen-google-objects-seq'
  'packing-unseen-google-objects-group'
  'put-block-in-bowl-full'
  'stack-block-pyramid-seq-full'
  'separating-piles-full'
  'towers-of-hanoi-seq-full'
)


# for task in "${tasks[@]}"
# do
#     echo "Running data colloection for task: $task"
#     python -m cliport.dataset_to_hdf5 \
#                             train.task="${task}"\
#                             train.n_demos=1000
# done


# for task in "${tasks[@]}"
# do
#     echo "Running data colloection for task: $task"
#     python cliport/demo_hdf5.py n=1000 \
#                             task=$task \
#                             mode=test \
#                             hdf5_path=extra2_dataset_no_aug
# done


echo "Running data colloection for task: multi-language-conditioned"
python -m cliport.dataset_to_hdf5_multi \
                        train.task="multi-language-conditioned"\
                        train.n_demos=1000 \
                        dataset.type=multi

