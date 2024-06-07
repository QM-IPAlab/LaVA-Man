#!/bin/bash
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --job-name=cliport
#SBATCH --cpus-per-task=16
#SBATCH --array=0-17%6
#SBATCH --time=48:00:00

module load python/anaconda3
source activate mae-cliport
export CLIPORT_ROOT=$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/mae

# ======== experiments name =======sb== #

exps_name="exps_all_0605_predict"


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
  "assembling-kits-seq-unseen-colors"\
  "assembling-kits-seq-seen-colors"\
  "packing-boxes-pairs-seen-colors"\
  "packing-boxes-pairs-unseen-colors"\
  "packing-seen-google-objects-group"\
  "packing-seen-google-objects-seq"\
  "packing-unseen-google-objects-group"\
  "packing-unseen-google-objects-seq"\
  "put-block-in-bowl-seen-colors"\
  "put-block-in-bowl-unseen-colors"\
  "separating-piles-seen-colors"\
  "separating-piles-unseen-colors"\
  "stack-block-pyramid-seq-seen-colors"\
  "stack-block-pyramid-seq-unseen-colors"\
  "towers-of-hanoi-seq-seen-colors"\
  "towers-of-hanoi-seq-unseen-colors" 
  #"packing-shapes"\
  #"align-rope"\
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
                         wandb.run_name=${agent_name}_${task_name} \
                         wandb.logger.group=${exps_name} \
                         pretrain_path=/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/output_mae_robot_lang_big_noref/checkpoint-160.pth

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

# python -m cliport.eval model_task=${task_name}\
#                        eval_task=${task_name} \
#                        agent=${agent_name} \
#                        mode=val \
#                        n_demos=100 \
#                        train_demos=100 \
#                        exp_folder=${exps_name} \
#                        checkpoint_type=val_missing \
#                        update_results=True \
#                        disp=False\
#                        record.save_video=False

python -m cliport.eval model_task=${task_name}\
                       eval_task=${task_name} \
                       agent=${agent_name} \
                       mode=test \
                       n_demos=100 \
                       train_demos=100 \
                       exp_folder=${exps_name} \
                       checkpoint_type=val_missing \
                       update_results=True \
                       disp=False\
                       record.save_video=False