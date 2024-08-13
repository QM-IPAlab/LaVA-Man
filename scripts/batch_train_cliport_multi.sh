#!/bin/bash
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --job-name=multi
#SBATCH --cpus-per-task=16

module load python/3.8
source py-mae-cliport/bin/activate
export CLIPORT_ROOT=$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/mae

exps_name="exps_all_extra_cos_lr"
#agent_name="transporter"
agent_name="mae_seg2"
#agent_name="rn50_bert"
#agent_name="clip_lingunet_transporter"

# ======== task name ========= #

# declare -a tasks=("assembling-kits-seq-seen-colors"\
#     "assembling-kits-seq-unseen-colors"\
#     "packing-boxes-pairs-seen-colors"\
#     "packing-boxes-pairs-unseen-colors"\
#     "stack-block-pyramid-seq-seen-colors"\
#     "stack-block-pyramid-seq-unseen-colors"\
#     "separating-piles-seen-colors"\
#     "separating-piles-unseen-colors"\
#     "towers-of-hanoi-seq-seen-colors"\
#     "towers-of-hanoi-seq-unseen-colors"\
#     "put-block-in-bowl-unseen-colors"\
#     "put-block-in-bowl-seen-colors"\
#     "packing-seen-google-objects-group"\
#     "packing-unseen-google-objects-group"\
#     "packing-seen-google-objects-seq"\
#     "packing-unseen-google-objects-seq"\
#     "align-rope"\
#     "packing-shapes"\
# )

declare -a tasks=("assembling-kits-seq-full"\
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

# python -m cliport.train  train.task=multi-language-conditioned\
#                          train.agent=${agent_name}\
#                          dataset.type=multi\
#                          train.n_demos=1000 \
#                          train.n_steps=80100 \
#                          train.exp_folder=${exps_name} \
#                          dataset.cache=True \
#                          train.load_from_last_ckpt=True \
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
#                          #train.warmup_epochs=10\



for task in "${tasks[@]}"
do
    echo "Running evaluation for agent: $agent with task: $task"
    python -m cliport.eval model_task=multi-language-conditioned\
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