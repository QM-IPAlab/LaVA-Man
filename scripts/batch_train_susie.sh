export CLIPORT_ROOT=$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/mae
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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

exps_name="exps_cliport/susie_0701"
agent_name="susie"
task_name="pack_objects"

# python -m cliport.train  train.task=${task_name}\
#                          train.agent=${agent_name}\
#                          train.exp_folder=${exps_name}\
#                          wandb.run_name=${exps_name}\
#                          train.n_demos=1000 \
#                          train.n_steps=20100 \
#                          train.lr_scheduler=True\
#                          train.lr=2e-5\
#                          train.warmup_epochs=10\
#                          train.precision=32\
#                          train.batch_size=8\
#                          train.batchnorm=True\
#                          train.load_from_last_ckpt=False\
#                          train.log=False\
#                          cliport_checkpoint=False\
#                          dataset.cache=False \
#                          train.sep_mode=pick\
#                          dataset.type=susie_real\


# python -m cliport.train  train.task=${task_name}\
#                          train.agent=${agent_name}\
#                          train.exp_folder=${exps_name}\
#                          wandb.run_name=${exps_name}\
#                          train.n_demos=1000 \
#                          train.n_steps=20100 \
#                          train.lr_scheduler=True\
#                          train.lr=2e-5\
#                          train.warmup_epochs=10\
#                          train.precision=32\
#                          train.batch_size=2\
#                          train.batchnorm=True\
#                          train.load_from_last_ckpt=False\
#                          train.log=False\
#                          cliport_checkpoint=False\
#                          dataset.cache=False \
#                          train.sep_mode=place\
#                          dataset.type=susie_real\
                         
# ======== Evaluation for simulated tasks ======== #
# python cliport/eval_sep.py model_task=put-block-in-bowl-full\
#                     eval_task=put-block-in-bowl-full\
#                     agent=${agent_name} \
#                     mode=test \
#                     n_demos=100 \
#                     train_demos=100 \
#                     exp_folder=${exps_name} \
#                     checkpoint_type=best \
#                     update_results=True \
#                     disp=False\
#                     record.save_video=False\


# ======== Evaluation for real tasks ======== #
python cliport/eval_pick_place_sep.py model_task=${task_name}\
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
                       type=susie_real\

python cliport/eval_pick_place_sep.py model_task=${task_name}\
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
                       type=susie_real\
                       