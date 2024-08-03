#!/bin/bash
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --job-name=pretrain
#SBATCH --cpus-per-task=16

module load python/anaconda3
source activate mae-cliport
export CLIPORT_ROOT=$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/mae

exps_name="exps_all_extra_cos_lr_lat"
agent_name="mae_seg2_lat"

# ======== task name ========= #

task_name="towers-of-hanoi-seq-unseen-colors"
#task_name="put-block-in-bowl-seen-colors"

#task_name="packing-seen-google-objects-group"
#task_name="packing-unseen-google-objects-group"

#task_name="packing-seen-google-objects-seq"
#task_name="packing-unseen-google-objects-seq"

# "packing-unseen-google-objects-seq"
# "towers-of-hanoi-seq-unseen-colors"
# "separating-piles-unseen-colors"
# "align-rope"
# "assembling-kits-seq-seen-colors"
# "assembling-kits-seq-unseen-colors"
# "packing-shapes"
# "packing-boxes-pairs-seen-colors"
# "packing-boxes-pairs-unseen-colors"
# "stack-block-pyramid-seq-seen-colors"
# "stack-block-pyramid-seq-unseen-colors"
# "separating-piles-seen-colors"
# "towers-of-hanoi-seq-seen-colors"
# "put-block-in-bowl-unseen-colors"
# "put-block-in-bowl-seen-colors"
# "packing-seen-google-objects-group"
# "packing-unseen-google-objects-group"
# "packing-seen-google-objects-seq"


# python -m cliport.train  train.task=${task_name}\
#                          train.agent=${agent_name}\
#                          train.n_demos=1000 \
#                          train.n_steps=60100 \
#                          train.exp_folder=${exps_name} \
#                          dataset.cache=True \
#                          train.load_from_last_ckpt=False \
#                          train.n_rotations=36\
#                          train.log=False \
#                          wandb.run_name=${exps_name}_${task_name} \
#                          mae_model=mae_robot_lang \
#                          train.linear_probe=False \
#                          train.accumulate_grad_batches=1 \
#                          pretrain_path=/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/output_mae_robot_lang_big_extra/checkpoint-140.pth\
#                          cliport_checkpoint=False\
#                          train.lr_scheduler=False\
#                          train.lr=1e-4\
#                          dataset.type=multi\
#                          #train.lr=5e-5\
#                          #train.warmup_epochs=10\
#                          #train.precision=32\
#                          #train.batch_size=4 \
                         


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
                      record.save_video=False\
                      #sep_model=True


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