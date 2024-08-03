# #!/bin/bash
# #SBATCH --partition=small
# #SBATCH --gres=gpu:1
# #SBATCH --job-name=real
# #SBATCH --cpus-per-task=16

# module load python/anaconda3
# source activate mae-cliport
export CLIPORT_ROOT=$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/mae

exps_name="exps_real"
agent_name="mae_seg2"

# ======== task name ========= #

task_name="pack_objects"

# python -m cliport.train  train.task=${task_name}\
#                          train.agent=${agent_name}\
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
#                          train.lr_scheduler=True\
#                          train.lr=5e-5\
#                          dataset.type=real\
#                          #train.lr=5e-5\
#                          #train.warmup_epochs=10\
#                          #train.precision=32\
#                          #train.batch_size=4 \
                         

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
                       type=real\


python cliport/eval_pick_place.py model_task=${task_name}\
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