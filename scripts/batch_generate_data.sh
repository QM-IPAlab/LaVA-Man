#!/bin/bash
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --job-name=gen_data
#SBATCH --cpus-per-task=16
module load python/anaconda3
source activate mae-cliport
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


declare -a tasks=(
    "towers-of-hanoi-seq-unseen-colors"
)

for task in "${tasks[@]}"
do
    echo "Running data colloection for task: $task"
    python -m cliport.dataset_to_hdf5 \
                            train.task="${task}"\
                            train.n_demos=100
done