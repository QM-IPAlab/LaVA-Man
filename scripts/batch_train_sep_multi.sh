#!/bin/bash
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --job-name=no_pretrain
#SBATCH --cpus-per-task=16

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
# 3. pretrain_path is correct
# 4. dataset.type is set to multi
# 5. train.sep_mode is set to pick or place
# 6. train.batchnorm is set to True
# 7. task name is multi-language-conditioned
# 8. wandb.run_name is set to exps_name_multi


exps_name="exps_no_pretrain"
agent_name="mae_sep_seg2"
#pretrain_path="/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/output_mae_robot_lang_big/checkpoint-160.pth"
#pretrain_path="/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/output_mae_robot_lang_big_m050/checkpoint-120.pth"
#pretrain_path="/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/output_mae_robot_lang_big_m075/checkpoint-120.pth"
#pretrain_path="/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/output_mae_robot_lang_big_m100/checkpoint-120.pth"
#pretrain_path="/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/output_mae_robot_lang_full_color/checkpoint-160.pth"
#pretrain_path="/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/output_mae_robot_lang_full_color/checkpoint-399.pth"
#pretrain_path="/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/output_mae_robot_lang_full_color/checkpoint-280.pth"
#pretrain_path="/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/output_mae_robot_lang_big/checkpoint-160.pth"
#pretrain_path="/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/output_mae_robot_lang_big/checkpoint-280.pth"
#pretrain_path="/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/output_mae_robot_lang_big/checkpoint-380.pth"
#pretrain_path="/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/output_mae_robot_lang_big_extra2/checkpoint-160.pth"
#pretrain_path="/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/output_mae_robot_lang_dual_masking/checkpoint-159.pth"
pretrain_path=False


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
    "align-rope"\
    "packing-shapes"\
)

python -m cliport.train  train.task=multi-language-conditioned\
                         train.agent=${agent_name}\
                         train.exp_folder=${exps_name}\
                         wandb.run_name=${exps_name}_multi\
                         train.n_demos=1000 \
                         train.n_steps=101000 \
                         train.lr_scheduler=True\
                         train.lr=2e-5\
                         train.warmup_epochs=10\
                         train.precision=32\
                         train.batch_size=32\
                         train.batchnorm=True\
                         train.load_from_last_ckpt=False\
                         train.log=True\
                         mae_model=mae_robot_lang \
                         pretrain_path=${pretrain_path}\
                         cliport_checkpoint=False\
                         dataset.cache=False \
                         train.sep_mode=pick\
                         dataset.type=multi\


python -m cliport.train  train.task=multi-language-conditioned\
                         train.agent=${agent_name}\
                         train.exp_folder=${exps_name}\
                         wandb.run_name=${exps_name}_multi\
                         train.n_demos=1000 \
                         train.n_steps=101000 \
                         train.lr_scheduler=True\
                         train.lr=2e-5\
                         train.warmup_epochs=10\
                         train.precision=32\
                         train.batch_size=16\
                         train.batchnorm=True\
                         train.load_from_last_ckpt=False\
                         train.log=True\
                         mae_model=mae_robot_lang \
                         pretrain_path=${pretrain_path}\
                         cliport_checkpoint=False\
                         dataset.cache=False \
                         train.sep_mode=place\
                         dataset.type=multi\

                       
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
                        checkpoint_type=test_best \
                        update_results=True \
                        disp=False\
                        record.save_video=False
done