export CLIPORT_ROOT=$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/mae
export CUDA_VISIBLE_DEVICES=1
# ======== experiments name =======sb== #

exps_name="exps_cliport/0419_cliport"
#agent_name="transporter"
agent_name="cliport"
#agent_name="rn50_bert"
#agent_name="clip_lingunet_transporter"

# ======== agent name ========= #


# python -m cliport.train  train.task=packing-omni-objects\
#                          train.agent=${agent_name}\
#                          train.exp_folder=${exps_name}\
#                          wandb.run_name=${exps_name}\
#                          train.n_demos=1000 \
#                          train.n_steps=60100 \
#                          train.load_from_last_ckpt=True\
#                          dataset.cache=True \
#                          train.load_pretrained_ckpt=False\
#                          train.sep_mode=False\
#                          train.data_dir="/home/robot/Repositories_chaoran/CLIPort_new_loss/data_ours"\                       
                         

# python -m cliport.eval model_task=packing-omni-objects\
#                        eval_task=packing-omni-objects \
#                        agent=${agent_name} \
#                        mode=val \
#                        n_demos=100 \
#                        train_demos=10 \
#                        exp_folder=${exps_name} \
#                        checkpoint_type=val_missing \
#                        update_results=True \
#                        disp=False\
#                        record.save_video=False


python -m cliport.eval model_task=packing-omni-objects\
                       eval_task=packing-omni-objects-group \
                       agent=${agent_name} \
                       mode=test \
                       n_demos=100 \
                       train_demos=1000 \
                       exp_folder=${exps_name} \
                       checkpoint_type=test_best \
                       update_results=True \
                       disp=False\
                       record.save_video=False\
                       data_dir="/home/robot/Repositories_chaoran/CLIPort_new_loss/data_ours"


python -m cliport.eval model_task=packing-omni-objects\
                       eval_task=packing-omni-objects-group-intra \
                       agent=${agent_name} \
                       mode=test \
                       n_demos=100 \
                       train_demos=1000 \
                       exp_folder=${exps_name} \
                       checkpoint_type=test_best \
                       update_results=True \
                       disp=False\
                       record.save_video=False\
                       data_dir="/home/robot/Repositories_chaoran/CLIPort_new_loss/data_ours" 


python -m cliport.eval model_task=packing-omni-objects\
                       eval_task=packing-omni-objects-group-inter \
                       agent=${agent_name} \
                       mode=test \
                       n_demos=100 \
                       train_demos=1000 \
                       exp_folder=${exps_name} \
                       checkpoint_type=test_best \
                       update_results=True \
                       disp=False\
                       record.save_video=False\
                       data_dir="/home/robot/Repositories_chaoran/CLIPort_new_loss/data_ours"

