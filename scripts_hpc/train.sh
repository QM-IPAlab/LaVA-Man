#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 12       # 12 cores (12 cores per GPU)
#$ -l h_rt=1:0:0    # 1 hour runtime (required to run on the short queue)
#$ -l h_vmem=7.5G   # 7.5 * 12 = 90G total RAM
#$ -l gpu=1         # request 1 GPU

module load anaconda3
source activate mae-cliport
export CLIPORT_ROOT=$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/mae
export TOKENIZERS_PARALLELISM=False

exps_name="exps_cliport_no_pretrain"
agent_name="mae_sep_dpt"
task_name="towers-of-hanoi-seq-seen-colors"

python -m cliport.train  train.task=${task_name}\
                         train.agent=${agent_name}\
                         train.exp_folder=${exps_name}\
                         train.n_demos=100 \
                         train.n_steps=20100 \
                         train.lr_scheduler=False\
                         train.lr=5e-5\
                         train.warmup_epochs=10\
                         train.precision=32\
                         train.batch_size=32\
                         train.load_from_last_ckpt=False\
                         train.log=False\
                         wandb.run_name=${exps_name}_${task_name}\
                         mae_model=mae_robot_lang \
                         pretrain_path=False\
                         cliport_checkpoint=False\
                         dataset.cache=True \
                         train.sep_mode=pick
                         #dataset.type=multi\