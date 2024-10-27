#!/bin/bash
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --job-name=extra2
#SBATCH --cpus-per-task=16
#SBATCH --array=0-11

module load python/3.8
source py-mae-cliport/bin/activate
export CLIPORT_ROOT=$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/mae
export TOKENIZERS_PARALLELISM=false

# ======== Checklist ========= #
# Check the following before running this script:
# 1. The job name above is correct !
# 2. The number of job array is 0 indexed


exps_name="exps_extra_10demos_loadMulti"
agent_name="mae_sep_seg2"

tasks=("assembling-kits-seq-full"\
    "packing-boxes-pairs-full"\
    "stack-block-pyramid-seq-full"\
    "separating-piles-full"\
    "towers-of-hanoi-seq-full"\
    "put-block-in-bowl-full"\
    "packing-seen-google-objects-group"\
    "packing-unseen-google-objects-group"\
    "packing-seen-google-objects-seq"\
    "packing-unseen-google-objects-seq"\
    #"align-rope"\
    #"packing-shapes"\
)

# tasks=("assembling-kits-seq-unseen-colors"\
#   "assembling-kits-seq-seen-colors"\
#   "towers-of-hanoi-seq-seen-colors"\
#   "towers-of-hanoi-seq-unseen-colors"\
#   "stack-block-pyramid-seq-seen-colors"\
#   "stack-block-pyramid-seq-unseen-colors"\
#   "separating-piles-seen-colors"\
#   "separating-piles-unseen-colors"\
#   "put-block-in-bowl-seen-colors"\
#   "put-block-in-bowl-unseen-colors"\
#   "packing-boxes-pairs-seen-colors"\
#   "packing-boxes-pairs-unseen-colors"\
#   "packing-seen-google-objects-group"\
#   "packing-seen-google-objects-seq"\
#   "packing-unseen-google-objects-group"\
#   "packing-unseen-google-objects-seq"\
#   # "packing-shapes"\
#   # "align-rope"\
# )

#/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/output_mae_robot_lang_big/checkpoint-160.pth

task_name=${tasks[$SLURM_ARRAY_TASK_ID]}
short_name=$(echo $task_name | awk -F '-' '{print $1 "-" $2 "-" $(NF-1)}')


python -m cliport.train  train.task=${task_name}\
                         train.agent=${agent_name}\
                         train.exp_folder=${exps_name}\
                         wandb.run_name=${exps_name}_place_${short_name}\
                         train.n_demos=10 \
                         train.n_steps=10100 \
                         train.lr_scheduler=True\
                         train.lr=1e-5\
                         train.warmup_epochs=10\
                         train.precision=32\
                         train.batch_size=16\
                         train.load_from_last_ckpt=True\
                         train.log=False\
                         mae_model=mae_robot_lang \
                         pretrain_path=/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/output_mae_robot_lang_big_extra/checkpoint-140.pth\
                         train.load_pretrained_ckpt=False\
                         dataset.cache=False \
                         train.sep_mode=place\
                         cliport_checkpoint=/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/exps_extra_seg2/multi-language-conditioned-mae_sep_seg2-n1000-train/checkpoints/place-best.ckpt\



python -m cliport.train  train.task=${task_name}\
                         train.agent=${agent_name}\
                         train.exp_folder=${exps_name}\
                         wandb.run_name=${exps_name}_pick_${short_name}\
                         train.n_demos=10 \
                         train.n_steps=10100 \
                         train.lr_scheduler=True\
                         train.lr=1e-5\
                         train.warmup_epochs=10\
                         train.precision=32\
                         train.batch_size=32\
                         train.load_from_last_ckpt=True\
                         train.log=False\
                         mae_model=mae_robot_lang \
                         pretrain_path=/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/output_mae_robot_lang_big_extra/checkpoint-140.pth\
                         train.load_pretrained_ckpt=False\
                         dataset.cache=False \
                         train.sep_mode=pick\
                         cliport_checkpoint=/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/exps_extra_seg2/multi-language-conditioned-mae_sep_seg2-n1000-train/checkpoints/pick-best.ckpt\


                       
python cliport/eval_sep.py model_task=${task_name}\
                      eval_task=${task_name} \
                      agent=${agent_name} \
                      mode=val \
                      n_demos=100 \
                      train_demos=10 \
                      exp_folder=${exps_name} \
                      checkpoint_type=val_missing \
                      update_results=True \
                      disp=False\
                      record.save_video=False\


python cliport/eval_sep.py model_task=${task_name}\
                      eval_task=${task_name} \
                      agent=${agent_name} \
                      mode=test \
                      n_demos=100 \
                      train_demos=10 \
                      exp_folder=${exps_name} \
                      checkpoint_type=test_best \
                      update_results=True \
                      disp=False\
                      record.save_video=False\