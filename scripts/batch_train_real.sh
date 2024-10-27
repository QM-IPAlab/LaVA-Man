#!/bin/bash
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --job-name=real
#SBATCH --cpus-per-task=16

module load python/3.8
source py-mae-cliport/bin/activate
export CLIPORT_ROOT=$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/mae

exps_name="exps_cliport_mix"
agent_name="cliport"

# ======== task name ========= #

task_name="pack_objects"

python cliport/train.py  train.task=multi-language-conditioned\
                         train.agent=${agent_name}\
                         train.exp_folder=${exps_name}\
                         wandb.run_name=debug\
                         train.n_demos=1000 \
                         train.n_steps=60100 \
                         train.precision=32\
                         train.batch_size=1\
                         train.batchnorm=True\
                         train.load_from_last_ckpt=False\
                         train.log=False\
                         cliport_checkpoint=False\
                         dataset.cache=False \
                         dataset.type=mix\

# python -m cliport.train  train.task=${task_name}\
#                          train.agent=${agent_name}\
#                          train.exp_folder=${exps_name} \
#                          dataset.cache=True \
#                          train.n_demos=100 \
#                          train.n_steps=20100 \
#                          train.load_from_last_ckpt=False \
#                          train.n_rotations=36\
#                          train.log=True \
#                          wandb.run_name=${exps_name}_${task_name} \
#                          mae_model=mae_robot_lang \
#                          train.linear_probe=False \
#                          train.accumulate_grad_batches=1 \
#                          pretrain_path=/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/output_mae_robot_lang_mix_v2/checkpoint-160.pth\
#                          train.lr_scheduler=True\
#                          train.lr=5e-5\
#                          dataset.type=real\
#                          train.warmup_epochs=3\
#                          train.load_pretrained_ckpt=False\
#                          cliport_checkpoint=/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/exps_extra_unbatched/multi-language-conditioned-mae_seg2-n1000-train/checkpoints/best.ckpt\


# python cliport/train.py  train.task=pack_objects\
#                          train.agent=${agent_name}\
#                          train.exp_folder=${exps_name}\
#                          wandb.run_name=exps_clip_nofreeze_real\
#                          train.n_demos=100 \
#                          train.n_steps=20100 \
#                          train.lr_scheduler=True\
#                          train.lr=5e-5\
#                          train.warmup_epochs=8\
#                          train.precision=32\
#                          train.batch_size=4\
#                          train.batchnorm=True\
#                          train.load_from_last_ckpt=False\
#                          train.log=False\
#                          mae_model=robot_clip \
#                          pretrain_path=/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/output_robot_clip_nofreeze/checkpoint-140.pth\
#                          cliport_checkpoint=False\
#                          dataset.cache=False \
#                          dataset.type=real\
#                          train.sep_mode=place

# python cliport/train.py  train.task=pack_objects\
#                          train.agent=${agent_name}\
#                          train.exp_folder=${exps_name}\
#                          wandb.run_name=exps_clip_nofreeze_real\
#                          train.n_demos=100 \
#                          train.n_steps=20100 \
#                          train.lr_scheduler=True\
#                          train.lr=5e-5\
#                          train.warmup_epochs=8\
#                          train.precision=32\
#                          train.batch_size=4\
#                          train.batchnorm=True\
#                          train.load_from_last_ckpt=False\
#                          train.log=False\
#                          mae_model=robot_clip \
#                          pretrain_path=/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/output_robot_clip_nofreeze/checkpoint-140.pth\
#                          cliport_checkpoint=False\
#                          dataset.cache=False \
#                          dataset.type=real\
#                          train.sep_mode=pick

# python cliport/eval_pick_place_sep.py model_task=pack_objects\
#                        eval_task=${task_name} \
#                        agent=${agent_name} \
#                        mode=test_unseen \
#                        n_demos=100 \
#                        train_demos=100 \
#                        exp_folder=${exps_name} \
#                        checkpoint_type=best \
#                        update_results=True \
#                        disp=False\
#                        record.save_video=False\
#                        type=real_ours\



python cliport/eval_pick_place.py model_task=${task_name}\
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
                       type=real_ours\


python cliport/eval_pick_place.py model_task=${task_name}\
                       eval_task=${task_name} \
                       agent=${agent_name} \
                       mode=test_unseen \
                       n_demos=100 \
                       train_demos=1000 \
                       exp_folder=${exps_name} \
                       checkpoint_type=best \
                       update_results=True \
                       disp=False\
                       record.save_video=False\
                       type=real\