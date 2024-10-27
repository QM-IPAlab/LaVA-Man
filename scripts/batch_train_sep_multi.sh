#!/bin/bash
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --job-name=mae_clip
#SBATCH --cpus-per-task=16

module load python/3.8
source py-mae-cliport/bin/activate
module load cuda/12.4
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


exps_name="exps_extra_mae_clip2"
agent_name="mae_sep_clip"
pretrain_path="/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/output_mae_clip_2/checkpoint-100.pth"
mae_model="mae_clip"
#pretrain_path=False

# tasks for ablation study (mask ratio)
# tasks=("assembling-kits-seq-seen-colors"
#   "towers-of-hanoi-seq-seen-colors"
#   "stack-block-pyramid-seq-seen-colors"
#   "separating-piles-seen-colors"
#   "put-block-in-bowl-seen-colors"
#   "packing-boxes-pairs-seen-colors"
#   "packing-seen-google-objects-group"
#   "packing-seen-google-objects-seq"
#   "packing-unseen-google-objects-group"
#   "packing-unseen-google-objects-seq"
# )

#tasks for testing
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

python -m cliport.train  train.task=multi-language-conditioned\
                         train.agent=${agent_name}\
                         train.exp_folder=${exps_name}\
                         wandb.run_name=${exps_name}_multi\
                         train.n_demos=1000 \
                         train.n_steps=101000 \
                         train.lr_scheduler=True\
                         train.lr=5e-5\
                         train.warmup_epochs=10\
                         train.precision=32\
                         train.batch_size=32\
                         train.batchnorm=True\
                         train.load_from_last_ckpt=False\
                         train.log=True\
                         mae_model=${mae_model} \
                         pretrain_path=${pretrain_path}\
                         cliport_checkpoint=False\
                         dataset.cache=False \
                         train.sep_mode=pick\
                         dataset.type=multi\
                         #text_model="openai/clip-vit-base-patch16"\
                         #train.linear_probe=True\


python -m cliport.train  train.task=multi-language-conditioned\
                         train.agent=${agent_name}\
                         train.exp_folder=${exps_name}\
                         wandb.run_name=${exps_name}_multi\
                         train.n_demos=1000 \
                         train.n_steps=101000 \
                         train.lr_scheduler=True\
                         train.lr=5e-5\
                         train.warmup_epochs=10\
                         train.precision=32\
                         train.batch_size=16\
                         train.batchnorm=True\
                         train.load_from_last_ckpt=False\
                         train.log=True\
                         mae_model=${mae_model} \
                         pretrain_path=${pretrain_path}\
                         cliport_checkpoint=False\
                         dataset.cache=False \
                         train.sep_mode=place\
                         dataset.type=multi\
                         #text_model="openai/clip-vit-base-patch16"\
                         #train.linear_probe=True\


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
    python cliport/eval_sep.py model_task=multi-language-conditioned\
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