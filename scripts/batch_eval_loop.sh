#!/bin/bash
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --job-name=testloop
#SBATCH --cpus-per-task=16

module load python/anaconda3
source activate mae-cliport
export CLIPORT_ROOT=$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/mae

# Define the experiment name
exps_name="exps_all_0601"

# Array of agent names
declare -a agents=("mae_seg2")

declare -a tasks=("assembling-kits-seq-seen-colors"\
    "assembling-kits-seq-unseen-colors"\
    "packing-boxes-pairs-seen-colors"\
    "packing-boxes-pairs-unseen-colors"\
    "stack-block-pyramid-seq-seen-colors"\
    "stack-block-pyramid-seq-unseen-colors"\
    "separating-piles-seen-colors"\
    "separating-piles-unseen-colors"\
    "towers-of-hanoi-seq-seen-colors"\
    "towers-of-hanoi-seq-unseen-colors"\
    "put-block-in-bowl-unseen-colors"\
    "put-block-in-bowl-seen-colors"\
    "packing-seen-google-objects-group"\
    "packing-unseen-google-objects-group"\
    "packing-seen-google-objects-seq"\
    "packing-unseen-google-objects-seq"\
)

# Loop over each agent
for agent in "${agents[@]}"
do
    # Loop over each task
    for task in "${tasks[@]}"
    do
        echo "Running evaluation for agent: $agent with task: $task"

        # Execute the Python script with the current agent and task
        python -m cliport.eval model_task=${task}\
                       eval_task=${task} \
                       agent=${agent} \
                       mode=val \
                       n_demos=100 \
                       train_demos=100 \
                       exp_folder=${exps_name} \
                       checkpoint_type=val_missing \
                       update_results=True \
                       disp=False\
                       record.save_video=False


        python -m cliport.eval model_task=${task}\
                            eval_task=${task} \
                            agent=${agent} \
                            mode=test \
                            n_demos=100 \
                            train_demos=100 \
                            exp_folder=${exps_name} \
                            checkpoint_type=test_best \
                            update_results=True \
                            disp=False\
                            record.save_video=False
    done
done
