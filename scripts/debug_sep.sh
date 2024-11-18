# Scripts for debug only ! 
# DO NOT SBATCH THIS SCRIPT !!!

export CLIPORT_ROOT=$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/mae
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1


agent_name="mae_sep_base"
task_name="pack_objects"

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


# python -m cliport.train  train.task=${task_name}\
#                          train.agent=${agent_name}\
#                          train.exp_folder=${exps_name}\
#                          wandb.run_name=${exps_name}_${task_name}\
#                          train.n_demos=100 \
#                          train.n_steps=20100 \
#                          train.lr_scheduler=True\
#                          train.lr=5e-5\
#                          train.warmup_epochs=10\
#                          train.load_from_last_ckpt=True\
#                          train.log=False\
#                          mae_model= \
#                          dataset.cache=True \
#                          pretrain_path=/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/output_mae_robot_lang_big_extra/checkpoint-140.pth\
#                          train.load_pretrained_ckpt=False\
#                          train.sep_mode=False\

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

# python cliport/train.py  train.task=multi-language-conditioned\
#                          train.agent=${agent_name}\
#                          train.exp_folder=debug\
#                          wandb.run_name=debug\
#                          train.n_demos=1000 \
#                          train.n_steps=101000 \
#                          train.lr_scheduler=True\
#                          train.lr=2e-5\
#                          train.warmup_epochs=10\
#                          train.precision=32\
#                          train.batch_size=16\
#                          train.batchnorm=True\
#                          train.load_from_last_ckpt=False\
#                          train.log=False\
#                          mae_model=mae_robot_lang \
#                          pretrain_path=/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/output_mae_robot_lang_mix_v2_full/checkpoint-60.pth\
#                          cliport_checkpoint=False\
#                          dataset.cache=False \
#                          train.sep_mode=place \
#                          dataset.type=multi\
#                          #text_model="openai/clip-vit-base-patch16"

# python cliport/train.py  train.task=pack_objects\
#                          train.agent=${agent_name}\
#                          train.exp_folder=${exps_name}\
#                          wandb.run_name=debug\
#                          train.n_demos=100 \
#                          train.n_steps=20100 \
#                          train.lr_scheduler=True\
#                          train.lr=2e-5\
#                          train.warmup_epochs=8\
#                          train.precision=32\
#                          train.batch_size=4\
#                          train.batchnorm=True\
#                          train.load_from_last_ckpt=False\
#                          train.log=False\
#                          mae_model=voltron \
#                          pretrain_path=/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/output_robot_clip_nofreeze/checkpoint-140.pth\
#                          cliport_checkpoint=False\
#                          dataset.cache=False \
#                          dataset.type=real\




# python cliport/train.py  train.task=multi-language-conditioned\
#                          train.agent=${agent_name}\
#                          train.exp_folder=${exps_name}\
#                          wandb.run_name=debug\
#                          train.n_demos=1000 \
#                          train.n_steps=60100 \
#                          train.precision=32\
#                          train.batch_size=1\
#                          train.batchnorm=True\
#                          train.load_from_last_ckpt=False\
#                          train.log=False\
#                          cliport_checkpoint=False\
#                          dataset.cache=False \
#                          dataset.type=mix\

    # python cliport/eval_sep.py model_task=multi-language-conditioned\
    #                     eval_task=packing-unseen-google-objects-seq \
    #                     agent=${agent_name} \
    #                     mode=test \
    #                     n_demos=100 \
    #                     train_demos=1000 \
    #                     exp_folder=exps_extra_sep_seg2_add \
    #                     checkpoint_type=best \
    #                     update_results=True \
    #                     disp=False\
    #                     record.save_video=False


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
                         train.batch_size=1\
                         train.batchnorm=True\
                         train.load_from_last_ckpt=False\
                         train.log=False\
                         mae_model=voltron \
                         pretrain_path=False\
                         cliport_checkpoint=False\
                         dataset.cache=False \
                         dataset.type=real_all\
                         train.sep_mode=pick\
                         train.linear_probe=True\
                         #text_model="openai/clip-vit-base-patch16"\


