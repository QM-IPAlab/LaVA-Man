# Scripts for debug only ! 
# DO NOT SBATCH THIS SCRIPT !!!

export CLIPORT_ROOT=$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/mae
export TOKENIZERS_PARALLELISM=false

exps_name="debug"
agent_name="mae_seg2"

task_name="put-block-in-bowl-seen-colors"

# "packing-unseen-google-objects-seq"
# "towers-of-hanoi-seq-unseen-colors"
# "separating-piles-unseen-colors"
# "align-rope"
# "assembling-kits-seq-seen-colors"
# "assembling-kits-seq-unseen-colors"
# "packing-shapes"
# "packing-boxes-pairs-seen-colors"
# "packing-boxes-pairs-unseen-colors"
# "stack-block-pyramid-seq-seen-colors"
# "stack-block-pyramid-seq-unseen-colors"
# "separating-piles-seen-colors"
# "towers-of-hanoi-seq-seen-colors"
# "put-block-in-bowl-unseen-colors"
# "put-block-in-bowl-seen-colors"
# "packing-seen-google-objects-group"
# "packing-unseen-google-objects-group"
# "packing-seen-google-objects-seq"


python -m cliport.train  train.task=${task_name}\
                         train.agent=${agent_name}\
                         train.exp_folder=${exps_name}\
                         wandb.run_name=${exps_name}_${task_name}\
                         train.n_demos=100 \
                         train.n_steps=20100 \
                         train.lr_scheduler=True\
                         train.lr=5e-5\
                         train.warmup_epochs=10\
                         train.load_from_last_ckpt=True\
                         train.log=True\
                         mae_model=mae_robot_lang \
                         dataset.cache=True \
                         pretrain_path=/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/output_mae_robot_lang_big_extra/checkpoint-140.pth\
                         train.load_pretrained_ckpt=False\
                         train.sep_mode=False\

# python -m cliport.train  train.task=${task_name}\
#                          train.agent=${agent_name}\
#                          train.exp_folder=${exps_name}\
#                          wandb.run_name=${exps_name}_place_${short_name}\
#                          train.n_demos=100 \
#                          train.n_steps=20100 \
#                          train.lr_scheduler=True\
#                          train.lr=5e-5\
#                          train.warmup_epochs=10\
#                          train.precision=32\
#                          train.batch_size=16\
#                          train.load_from_last_ckpt=True\
#                          train.log=True\
#                          mae_model=mae_robot_lang \
#                          pretrain_path=/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/output_mae_robot_lang_big_extra/checkpoint-140.pth\
#                          train.load_pretrained_ckpt=True\
#                          cliport_checkpoint=/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/exps_extra_seg2/multi-language-conditioned-mae_sep_seg2-n1000-train/checkpoints/place-best.ckpt\
#                          dataset.cache=False \
#                          train.sep_mode=place\


# python -m cliport.train  train.task=${task_name}\
#                          train.agent=${agent_name}\
#                          train.exp_folder=${exps_name}\
#                          wandb.run_name=debug\
#                          train.n_demos=100 \
#                          train.n_steps=20100 \
#                          train.lr_scheduler=True\
#                          train.lr=1e-5\
#                          train.warmup_epochs=10\
#                          train.precision=32\
#                          train.batch_size=16\
#                          train.load_from_last_ckpt=False\
#                          train.log=False\
#                          mae_model=mae_robot_lang \
#                          pretrain_path=/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/output_mae_robot_lang_big_extra/checkpoint-140.pth\
#                          cliport_checkpoint=False\
#                          dataset.cache=False \
#                          train.sep_mode=place\
#                          #dataset.type=multi\

# python -m cliport.train  train.task=multi-language-conditioned\
#                          train.agent=${agent_name}\
#                          train.exp_folder=${exps_name}\
#                          wandb.run_name=debug\
#                          train.n_demos=1000 \
#                          train.n_steps=101000 \
#                          train.lr_scheduler=True\
#                          train.lr=2e-5\
#                          train.warmup_epochs=8\
#                          train.precision=32\
#                          train.batch_size=16\
#                          train.batchnorm=True\
#                          train.load_from_last_ckpt=False\
#                          train.log=False\
#                          mae_model=mae_robot_lang \
#                          pretrain_path=/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/output_mae_robot_lang_big/checkpoint-160.pth\
#                          cliport_checkpoint=False\
#                          dataset.cache=False \
#                          train.sep_mode=pick\
#                          dataset.type=multi\

                         