export CLIPORT_ROOT=$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/mae
export TOKENIZERS_PARALLELISM=false

exps_name="debug"
agent_name="mae_sep_seg2_add"
pretrain_path="/data/home/acw694/CLIPort_new_loss/exps/multi_size/checkpoint-100.pth"
mae_model="mae_robot_lang"
#pretrain_path=False

# tasks for ablation study (mask ratio)
# tasks=("assembling-kits-seq-seen-colors"
#   "towers-of-hanoi-seq-seen-colors"
#   "stack-block-pyramid-seq-seen-colors"
#   "separating-piles-seen-colors"
#   "put-block-in-bowl-seen-colors"
#   "packing-boxes-pairs-seen-colors"
#   "packing-seen-google-objects-group"
#   "packing-seen-google-objects-seq"
#   "packing-unseen-google-objects-group"
#   "packing-unseen-google-objects-seq"
# )

#tasks for testing
tasks=("packing-unseen-google-objects-group"\
    "packing-seen-google-objects-seq"\
    "packing-unseen-google-objects-seq"\
    "separating-piles-full"\
    #"align-rope"\
    #"packing-shapes"\
)

python -m cliport.train  train.task=multi-language-conditioned\
                         train.agent=${agent_name}\
                         train.exp_folder=${exps_name}\
                         wandb.run_name=${exps_name}_multi\
                         train.n_demos=1000 \
                         train.n_steps=60100 \
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
                         train.sep_mode=pick\
                         dataset.type=multi\
                         train.linear_probe=True\
