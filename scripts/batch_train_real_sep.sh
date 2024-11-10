#!/bin/bash
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --job-name=clip_mae
#SBATCH --cpus-per-task=16

module load python/3.8
source py-mae-cliport/bin/activate
export CLIPORT_ROOT=$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/mae

exps_name="exps_real_all_sep_seg2_add_clipv"
agent_name="mae_sep_seg2_add_clipv"

pretrain_path="/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/output_mae_robot_lang_mix_v2_full/checkpoint-60.pth"

mae_model="mae_robot_lang"
task_name="pack_objects"

python cliport/train.py  train.task=${task_name}\
                         train.agent=${agent_name}\
                         train.exp_folder=${exps_name}\
                         wandb.run_name=${exps_name}_${task_name}\
                         train.n_demos=100 \
                         train.n_steps=40100 \
                         train.lr_scheduler=True\
                         train.lr=2e-5\
                         train.warmup_epochs=10\
                         train.precision=32\
                         train.batch_size=16\
                         train.batchnorm=True\
                         train.load_from_last_ckpt=False\
                         train.log=False\
                         mae_model=${mae_model}\
                         pretrain_path=${pretrain_path}\
                         cliport_checkpoint=False\
                         dataset.cache=False \
                         dataset.type=real_all\
                         train.sep_mode=place\
                         train.linear_probe=False\
                         #text_model="openai/clip-vit-base-patch16"\

python cliport/train.py  train.task=${task_name}\
                         train.agent=${agent_name}\
                         train.exp_folder=${exps_name}\
                         wandb.run_name=${exps_name}_${task_name}\
                         train.n_demos=100 \
                         train.n_steps=40100 \
                         train.lr_scheduler=True\
                         train.lr=2e-5\
                         train.warmup_epochs=10\
                         train.precision=32\
                         train.batch_size=32\
                         train.batchnorm=True\
                         train.load_from_last_ckpt=False\
                         train.log=False\
                         mae_model=${mae_model} \
                         pretrain_path=${pretrain_path}\
                         cliport_checkpoint=False\
                         dataset.cache=False \
                         dataset.type=real_all\
                         train.sep_mode=pick\
                         train.linear_probe=False\
                         #text_model="openai/clip-vit-base-patch16"\

python cliport/eval_pick_place_sep.py model_task=${task_name}\
                       eval_task=${task_name} \
                       agent=${agent_name} \
                       mode=test_unseen \
                       n_demos=100 \
                       train_demos=100 \
                       exp_folder=${exps_name} \
                       checkpoint_type=best \
                       update_results=True \
                       disp=False\
                       record.save_video=False\
                       type=real_ours\

python cliport/eval_pick_place_sep.py model_task=${task_name}\
                       eval_task=${task_name} \
                       agent=${agent_name} \
                       mode=test_unseen \
                       n_demos=100 \
                       train_demos=100 \
                       exp_folder=${exps_name} \
                       checkpoint_type=best \
                       update_results=True \
                       disp=False\
                       record.save_video=False\
                       type=real\

python cliport/eval_pick_place_sep.py model_task=${task_name}\
                       eval_task=${task_name} \
                       agent=${agent_name} \
                       mode=test_seen \
                       n_demos=100 \
                       train_demos=100 \
                       exp_folder=${exps_name} \
                       checkpoint_type=best \
                       update_results=True \
                       disp=False\
                       record.save_video=False\
                       type=real\