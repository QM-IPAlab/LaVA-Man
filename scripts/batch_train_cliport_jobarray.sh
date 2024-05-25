#!/bin/bash
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --job-name=seg_399
#SBATCH --cpus-per-task=16
#SBATCH --array=0-17%9
#SBATCH --time=24:00:00

module load python/anaconda3
source activate mae-cliport
export CLIPORT_ROOT=$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/mae

# ======== experiments name =======sb== #

exps_name="exps_mae_fix_pos_ck399"


# ======== agent name ========= #

agent_name="mae_seg2"

# tasks=(
#   "align-rope"
#   "assembling-kits-seq-seen-colors"
#   "assembling-kits-seq-unseen-colors"
#   "packing-shapes"
#   "packing-boxes-pairs-seen-colors"
#   "packing-boxes-pairs-unseen-colors"
#   "stack-block-pyramid-seq-seen-colors"
#   "stack-block-pyramid-seq-unseen-colors"
#   "separating-piles-seen-colors"
#   "separating-piles-unseen-colors"
#   "towers-of-hanoi-seq-seen-colors"
#   "towers-of-hanoi-seq-unseen-colors"
# )

tasks=(
  "packing-unseen-google-objects-seq"
  "towers-of-hanoi-seq-unseen-colors"
  #"separating-piles-unseen-colors"
  #"align-rope"
  "assembling-kits-seq-seen-colors"
  "assembling-kits-seq-unseen-colors"
  #"packing-shapes"
  #"packing-boxes-pairs-seen-colors"
  #"packing-boxes-pairs-unseen-colors"
  "stack-block-pyramid-seq-seen-colors"
  "stack-block-pyramid-seq-unseen-colors"
  #"separating-piles-seen-colors"
  "towers-of-hanoi-seq-seen-colors"
  "put-block-in-bowl-unseen-colors"
  "put-block-in-bowl-seen-colors"
  #"packing-seen-google-objects-group"
  #"packing-unseen-google-objects-group"
  "packing-seen-google-objects-seq"
)

task_name=${tasks[$SLURM_ARRAY_TASK_ID]}


python -m cliport.train  train.task=${task_name}\
                         train.agent=${agent_name}\
                         train.n_demos=100 \
                         train.n_steps=20100 \
                         train.exp_folder=${exps_name} \
                         dataset.cache=True \
                         train.load_from_last_ckpt=False \
                         train.n_rotations=36\
                         train.log=True \
                         wandb.run_name=${agent_name}_${exps_name} \
                         wandb.logger.tags=${task_name} \
                         pretrain_path=/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/output_mae_robot_lang2/checkpoint-399.pth


python -m cliport.eval model_task=${task_name}\
                       eval_task=${task_name} \
                       agent=${agent_name} \
                       mode=test \
                       n_demos=100 \
                       train_demos=100 \
                       exp_folder=${exps_name} \
                       checkpoint_type=best \
                       update_results=True \
                       disp=False\
                       record.save_video=False