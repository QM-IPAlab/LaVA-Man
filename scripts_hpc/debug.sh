export CLIPORT_ROOT=$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/mae

exps_name="debug"
agent_name="mae_sep_seg2"


# ======== task name ========= #

task_name="towers-of-hanoi-seq-seen-colors"
#task_name="put-block-in-bowl-seen-colors"

#task_name="packing-seen-google-objects-group"
#task_name="packing-unseen-google-objects-group"

#task_name="packing-seen-google-objects-seq"
#task_name="packing-unseen-google-objects-seq"


python -m cliport.train  train.task=${task_name}\
                         train.agent=${agent_name}\
                         train.n_demos=100 \
                         train.n_steps=20100 \
                         train.exp_folder=${exps_name} \
                         dataset.cache=True \
                         train.load_from_last_ckpt=False \
                         train.n_rotations=36\
                         train.log=False \
                         wandb.run_name=${exps_name}_${task_name} \
                         mae_model=mae_robot_lang \
                         pretrain_path=/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/output_mae_robot_lang_big_extra/checkpoint-140.pth\
                         cliport_checkpoint=False\
                         train.lr_scheduler=True\
                         train.lr=1e-4\
                         train.batch_size=4\
                         train.sep_mode=pick
                         #dataset.type=multi\
                         #train.lr=5e-5\
                         #train.warmup_epochs=10\
                         #train.precision=32\
                         #train.batch_size=4 \