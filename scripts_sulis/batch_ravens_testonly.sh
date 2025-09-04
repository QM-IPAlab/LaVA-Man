#!/bin/bash
#SBATCH --job-name=test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=42
#SBATCH --mem-per-cpu=3850
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --account=su008-acw694
#SBATCH --mail-type=ALL
#SBATCH --mail-user=chaoran.zhu@qmul.ac.uk

module load Miniconda3/4.12.0
source activate mae-cliport

export CLIPORT_ROOT=$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/mae
export TOKENIZERS_PARALLELISM=false

agent_name="mae_sep_base"
exps_name="exps_cliport/0429_mpi_omni"
train_demos=1000
mae_model="mpi"

#tasks for testing
tasks=("assembling-kits-seq-full"\
    "packing-boxes-pairs-full"\
    "stack-block-pyramid-seq-full"\
    "towers-of-hanoi-seq-full"\
    "put-block-in-bowl-full"\
    "packing-seen-google-objects-group"\
    "packing-unseen-google-objects-group"\
    "packing-seen-google-objects-seq"\
    "packing-unseen-google-objects-seq"\
    "separating-piles-full"\
)
                         
for task in "${tasks[@]}"
do
    echo "Running evaluation for agent: $agent with task: $task"
    python cliport/eval_sep.py model_task=packing-omni-objects\
                        eval_task=${task} \
                        agent=${agent_name} \
                        mode=test \
                        n_demos=100 \
                        train_demos=${train_demos} \
                        exp_folder=${exps_name} \
                        checkpoint_type=best\
                        update_results=True \
                        disp=False\
                        record.save_video=False

    python cliport/eval_sep.py model_task=packing-omni-objects\
                        eval_task=${task} \
                        agent=${agent_name} \
                        mode=test \
                        n_demos=100 \
                        train_demos=${train_demos} \
                        exp_folder=${exps_name} \
                        checkpoint_type=last\
                        update_results=True \
                        disp=False\
                        record.save_video=False
done