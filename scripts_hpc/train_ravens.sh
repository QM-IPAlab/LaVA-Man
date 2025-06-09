#!/bin/bash
#$ -cwd
#$ -j y
#$ -l h_rt=24:0:0
#$ -l h_vmem=7.5G
#$ -l gpu=2
#$ -l gpu_type=ampere
#$ -N pretrain
#$ -m bea
#$ -l rocky
#$ -pe smp 16     


module load miniforge
module load gcc/12.2.0
mamba activate mae-cliport


export CLIPORT_ROOT=$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/mae
export TOKENIZERS_PARALLELISM=false
#export CUDA_VISIBLE_DEVICES=0

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


exps_name="exps_cliport/0428_voltron"
agent_name="mae_sep_base"
pretrain_path="/data/home/acw694/CLIPort_new_loss/checkpoints/voltron-omni-checkpoint-399.pth"
mae_model="voltron"
#pretrain_path=False

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
    "align-rope"\
    "packing-shapes"\
)

python -m cliport.train  train.task=multi-language-conditioned-full\
                         train.agent=${agent_name}\
                         train.exp_folder=${exps_name}\
                         wandb.run_name=${exps_name}_multi\
                         train.n_demos=1000 \
                         train.n_steps=60100 \
                         train.lr_scheduler=True\
                         train.lr=2e-5\
                         train.warmup_epochs=10\
                         train.precision=32\
                         train.batch_size=32\
                         train.batchnorm=True\
                         train.load_from_last_ckpt=False\
                         train.log=False\
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
                         train.warmup_epochs=10\
                         train.precision=32\
                         train.batch_size=2\
                         train.batchnorm=True\
                         train.load_from_last_ckpt=False\
                         train.log=False\
                         mae_model=${mae_model} \
                         pretrain_path=${pretrain_path}\
                         cliport_checkpoint=False\
                         dataset.cache=False \
                         train.sep_mode=place\
                         dataset.type=multi\
                         train.linear_probe=False\
                         

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
                        checkpoint_type=best \
                        update_results=True \
                        disp=False\
                        record.save_video=False
done