#!/bin/bash
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --job-name=10lp
#SBATCH --cpus-per-task=16
#SBATCH --array=0-4

module load python/3.8
source py-mae-cliport/bin/activate
export CLIPORT_ROOT=$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/mae
export TOKENIZERS_PARALLELISM=false

# linear probe benchmark for 10 demos

exps_name="exps_extra_sep_seg2_fm_10demoslp"
agent_name="mae_sep_seg2_fm"

tasks=("stack-block-pyramid-seq-full"\
    "packing-seen-google-objects-seq"\
    "packing-seen-google-objects-group"\
    "packing-unseen-google-objects-seq"\
    "packing-unseen-google-objects-group"\  
    "assembling-kits-seq-full"\
    "towers-of-hanoi-seq-full"\
    "put-block-in-bowl-full"\
    "separating-piles-full"\
    "packing-boxes-pairs-full"\
    "align-rope"\
    "packing-shapes"\
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
                         train.n_steps=20100 \
                         train.lr_scheduler=True\
                         train.lr=2e-5\
                         train.warmup_epochs=10\
                         train.precision=32\
                         train.batch_size=16\
                         train.load_from_last_ckpt=True\
                         train.log=False\
                         mae_model=mae_robot_lang \
                         pretrain_path=/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/output_mae_robot_lang_big_extra2/checkpoint-160.pth\
                         train.load_pretrained_ckpt=False\
                         dataset.cache=False \
                         dataset.aug=True \
                         train.sep_mode=place\
                         train.linear_probe=True
                         #cliport_checkpoint=/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/exps_extra_seg2/multi-language-conditioned-mae_sep_seg2-n1000-train/checkpoints/place-best.ckpt\



python -m cliport.train  train.task=${task_name}\
                         train.agent=${agent_name}\
                         train.exp_folder=${exps_name}\
                         wandb.run_name=${exps_name}_pick_${short_name}\
                         train.n_demos=10 \
                         train.n_steps=20100 \
                         train.lr_scheduler=True\
                         train.lr=2e-5\
                         train.warmup_epochs=10\
                         train.precision=32\
                         train.batch_size=32\
                         train.load_from_last_ckpt=True\
                         train.log=False\
                         mae_model=mae_robot_lang \
                         pretrain_path=/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/output_mae_robot_lang_big_extra2/checkpoint-160.pth\
                         train.load_pretrained_ckpt=False\
                         dataset.cache=False \
                         dataset.aug=True \
                         train.sep_mode=pick\
                         train.linear_probe=True
                         #cliport_checkpoint=/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/exps_extra_seg2/multi-language-conditioned-mae_sep_seg2-n1000-train/checkpoints/pick-best.ckpt\


                       
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