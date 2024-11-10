#!/bin/bash
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --job-name=1demo
#SBATCH --cpus-per-task=16
#SBATCH --array=0-9

module load python/3.8
source py-mae-cliport/bin/activate
export CLIPORT_ROOT=$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/mae

# ======== experiments name =======sb== #

exps_name="exps_extra_1demos_cliport"


# ======== agent name ========= #

agent_name="cliport"
#agent_name="transporter"
#agent_name="cliport"
#agent_name="rn50_bert"
#agent_name="clip_lingunet_transporter"

# tasks=("packing-unseen-google-objects-group"\
#   "packing-unseen-google-objects-seq"\
#   "packing-seen-google-objects-seq"\
# )

tasks=("assembling-kits-seq-full"\
    "packing-boxes-pairs-full"\
    "stack-block-pyramid-seq-full"\
    "separating-piles-full"\
    "towers-of-hanoi-seq-full"\
    "put-block-in-bowl-full"\
    "packing-seen-google-objects-group"\
    "packing-unseen-google-objects-group"\
    "packing-seen-google-objects-seq"\
    "packing-unseen-google-objects-seq")

#/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/output_mae_robot_lang_big/checkpoint-160.pth

task_name=${tasks[$SLURM_ARRAY_TASK_ID]}

python -m cliport.train  train.task=${task_name}\
                         train.agent=${agent_name}\
                         train.exp_folder=${exps_name}\
                         wandb.run_name=${exps_name}_${task_name}\
                         train.n_demos=1 \
                         train.n_steps=20100 \
                         train.load_from_last_ckpt=True\
                         dataset.cache=True \
                         train.load_pretrained_ckpt=False\
                         train.sep_mode=False\
                         #cliport_checkpoint=/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/exps_cliport_pretrained/multi-language-conditioned-${agent_name}-n1000-train/checkpoints/best.ckpt\
                         
                         

python -m cliport.eval model_task=${task_name}\
                       eval_task=${task_name} \
                       agent=${agent_name} \
                       mode=val \
                       n_demos=100 \
                       train_demos=1 \
                       exp_folder=${exps_name} \
                       checkpoint_type=val_missing \
                       update_results=True \
                       disp=False\
                       record.save_video=False


python -m cliport.eval model_task=${task_name}\
                       eval_task=${task_name} \
                       agent=${agent_name} \
                       mode=test \
                       n_demos=100 \
                       train_demos=1 \
                       exp_folder=${exps_name} \
                       checkpoint_type=test_best \
                       update_results=True \
                       disp=False\
                       record.save_video=False