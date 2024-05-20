#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --job-name=cliport
#SBATCH --cpus-per-task=16
module load python/anaconda3
source activate mae-cliport
export CLIPORT_ROOT=$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/mae

exps_name="exps_mae_rl2"
agent_name="mae"

# ======== task name ========= #

#task_name="put-block-in-bowl-unseen-colors"
#task_name="put-block-in-bowl-seen-colors"

#task_name="packing-seen-google-objects-group"
#task_name="packing-unseen-google-objects-group"

#task_name="packing-seen-google-objects-seq"
task_name="packing-unseen-google-objects-seq"


python -m cliport.train  train.task=${task_name}\
                         train.agent=${agent_name}\
                         train.attn_stream_fusion_type=add \
                         train.trans_stream_fusion_type=conv \
                         train.lang_fusion_type=mult \
                         train.n_demos=100 \
                         train.n_steps=20100 \
                         train.exp_folder=${exps_name} \
                         dataset.cache=True \
                         train.load_from_last_ckpt=False \
                         train.n_rotations=36\
                         train.log=False \


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