#!/bin/bash
#SBATCH --job-name=fuse
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=42
#SBATCH --mem-per-cpu=3850
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --account=su008-acw694
#SBATCH --mail-type=ALL
#SBATCH --mail-user=chaoran.zhu@qmul.ac.uk

module load Miniconda3/4.12.0
source activate mae-cliport

export CLIPORT_ROOT=$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/mae
export TOKENIZERS_PARALLELISM=false

# ======== Checklist ========= #
# Check the following before running this script:
# 1. The job name above is correct !
# 2. The number of job array is 0 indexed
# 3. pretrain_path is correct
# 4. dataset.type is set to multi
# 5. train.batchnorm is set to True
# 6. task name is multi-language-conditioned
# 7. wandb.run_name is set to exps_name_multi
# 8. check the agent name: sep or not sept, if sep, check train.sep_mode is set to pick or place


exps_name="exps_cliport/0430_mpi"
agent_name="mae_sep_base"
pretrain_path="/home/a/acw694/CLIPort_new_loss/checkpoints/MPI-base-state_dict.pt"
mae_model="mpi"
#pretrain_path=False

# #tasks for ablation study (mask ratio)
# tasks=("assembling-kits-seq-seen-colors"
#   "assembling-kits-seq-unseen-colors"
#   "towers-of-hanoi-seq-seen-colors"
#   "towers-of-hanoi-seq-unseen-colors"
#   "stack-block-pyramid-seq-seen-colors"
#   "stack-block-pyramid-seq-unseen-colors"
#   "separating-piles-seen-colors"
#   "separating-piles-unseen-colors"
#   "put-block-in-bowl-seen-colors"
#   "put-block-in-bowl-unseen-colors"
#   "packing-boxes-pairs-seen-colors"
#   "packing-boxes-pairs-unseen-colors"
#   "packing-seen-google-objects-group"
#   "packing-seen-google-objects-seq"
#   "packing-/home/a/acw694/CLIPort_new_loss/exps_cliport/0412_multisize-ck220/multi-language-conditioned-mae_fuse-n100-trainunseen-google-objects-group"
#   "packing-unseen-google-objects-seq"
#   "align-rope"
#   "packing-shapes"
# )

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

python -m cliport.train  train.task=multi-language-conditioned-full\
                         train.agent=${agent_name}\
                         train.exp_folder=${exps_name}\
                         wandb.run_name=${exps_name}_multi\
                         train.n_demos=1000 \
                         train.n_steps=60100 \
                         train.lr_scheduler=True\
                         train.lr=2e-5\
                         train.warmup_epochs=20\
                         train.precision=32\
                         train.batch_size=16\
                         train.batchnorm=True\
                         train.load_from_last_ckpt=False\
                         train.log=True\
                         mae_model=${mae_model} \
                         pretrain_path=${pretrain_path}\
                         cliport_checkpoint=False\
                         dataset.cache=False \
                         train.sep_mode=pick\
                         dataset.type=multi\
                         train.linear_probe=False\


python -m cliport.train  train.task=multi-language-conditioned-full\
                         train.agent=${agent_name}\
                         train.exp_folder=${exps_name}\
                         wandb.run_name=${exps_name}_multi\
                         train.n_demos=1000 \
                         train.n_steps=60100 \
                         train.lr_scheduler=True\
                         train.lr=2e-5\
                         train.warmup_epochs=20\
                         train.precision=32\
                         train.batch_size=4\
                         train.batchnorm=True\
                         train.load_from_last_ckpt=False\
                         train.log=True\
                         mae_model=${mae_model} \
                         pretrain_path=${pretrain_path}\
                         cliport_checkpoint=False\
                         dataset.cache=False \
                         train.sep_mode=place\
                         dataset.type=multi\
                         train.linear_probe=False\
                         

# python cliport/eval_pick_place_sep.py model_task=multi-language-conditioned\
#                        eval_task=pack_objects \
#                        agent=${agent_name} \
#                        mode=test_unseen \
#                        n_demos=100 \
#                        train_demos=1000 \
#                        exp_folder=${exps_name} \
#                        checkpoint_type=best \
#                        update_results=True \
#                        disp=False\
#                        record.save_video=False\
#                        type=real\

# python cliport/eval_pick_place_sep.py model_task=multi-language-conditioned\
#                        eval_task=pack_objects \
#                        agent=${agent_name} \
#                        mode=test_seen \
#                        n_demos=100 \
#                        train_demos=1000 \
#                        exp_folder=${exps_name} \
#                        checkpoint_type=best \
#                        update_results=True \
#                        disp=False\
#                        record.save_video=False\
#                        type=real\

for task in "${tasks[@]}"
do
    echo "Running evaluation for agent: $agent with task: $task"
    python cliport/eval_sep.py model_task=multi-language-conditioned-full\
                        eval_task=${task} \
                        agent=${agent_name} \
                        mode=test \
                        n_demos=100 \
                        train_demos=1000 \
                        exp_folder=${exps_name} \
                        checkpoint_type=best\
                        update_results=True \
                        disp=False\
                        record.save_video=False

    # python cliport/eval_sep.py model_task=multi-language-conditioned-full\
    #                     eval_task=${task} \
    #                     agent=${agent_name} \
    #                     mode=test \
    #                     n_demos=100 \
    #                     train_demos=1000 \
    #                     exp_folder=${exps_name} \
    #                     checkpoint_type=last\
    #                     update_results=True \
    #                     disp=False\
    #                     record.save_video=False
done