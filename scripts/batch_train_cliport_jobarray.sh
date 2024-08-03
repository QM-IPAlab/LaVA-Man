#!/bin/bash
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --job-name=dpt2loss
#SBATCH --cpus-per-task=16
#SBATCH --array=0-5

module load python/anaconda3
source activate mae-cliport
export CLIPORT_ROOT=$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/mae

# ======== experiments name =======sb== #

exps_name="exps_all_extra_cos_lr_dpt_2loss"


# ======== agent name ========= #

agent_name="mae_seg_dpt_2loss"

# tasks=("packing-unseen-google-objects-group"\
#   "packing-unseen-google-objects-seq"\
#   "packing-seen-google-objects-seq"\
# )

tasks=("assembling-kits-seq-unseen-colors"\
  "assembling-kits-seq-seen-colors"\
  "towers-of-hanoi-seq-seen-colors"\
  "towers-of-hanoi-seq-unseen-colors"\
  "stack-block-pyramid-seq-seen-colors"\
  "stack-block-pyramid-seq-unseen-colors"\
  #"separating-piles-seen-colors"\
  #"separating-piles-unseen-colors"\
  #"put-block-in-bowl-seen-colors"\
  #"put-block-in-bowl-unseen-colors"\
  #"packing-boxes-pairs-seen-colors"\
  #"packing-boxes-pairs-unseen-colors"\
  #"packing-seen-google-objects-group"\
  #"packing-seen-google-objects-seq"\
  #"packing-unseen-google-objects-group"\
  #"packing-unseen-google-objects-seq"\
  #"packing-shapes"\
  #"align-rope"\
)

#/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/output_mae_robot_lang_big/checkpoint-160.pth

task_name=${tasks[$SLURM_ARRAY_TASK_ID]}


# python -m cliport.train  train.task=${task_name}\
#                          train.agent=${agent_name}\
#                          train.n_demos=100 \
#                          train.n_steps=20100 \
#                          train.exp_folder=${exps_name} \
#                          dataset.cache=True \
#                          train.load_from_last_ckpt=False \
#                          train.n_rotations=36\
#                          train.log=True \
#                          wandb.run_name=${exps_name}_${task_name} \
#                          mae_model=mae_robot_lang \
#                          train.linear_probe=False \
#                          train.accumulate_grad_batches=1 \
#                          pretrain_path=/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/output_mae_robot_lang_big_extra/checkpoint-140.pth\
#                          cliport_checkpoint=False\
#                          train.lr_scheduler=True\
#                          train.lr=5e-5\
#                          train.warmup_epochs=10\


python -m cliport.eval model_task=${task_name}\
                       eval_task=${task_name} \
                       agent=${agent_name} \
                       mode=val \
                       n_demos=100 \
                       train_demos=100 \
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
                       train_demos=100 \
                       exp_folder=${exps_name} \
                       checkpoint_type=test_best \
                       update_results=True \
                       disp=False\
                       record.save_video=False