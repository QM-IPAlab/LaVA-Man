#!/bin/bash
#SBATCH --job-name=fuse
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=
#SBATCH --cpus-per-task=42
#SBATCH --mem-per-cpu=3850
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --account=su008-acw694
#SBATCH --mail-type=ALL
#SBATCH --mail-user=chaoran.zhu@qmul.ac.uk


# ====== Usage ======
# Scripts for training and testing on downstream task of ravens dataset

#module load Miniconda3/4.12.0
#source activate mae-cliport

export CLIPORT_ROOT=$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/mae
export TOKENIZERS_PARALLELISM=false

exps_name="exps_cliport/0426_mpi_omini"
agent_name="mae_sep_base"
pretrain_path="/home/a/acw694/CLIPort_new_loss/exps/0425_mpi/checkpoint-399.pth"
mae_model="mpi"

python -m cliport.train  train.task=packing-omni-objects\
                         train.agent=${agent_name}\
                         train.exp_folder=${exps_name}\
                         wandb.run_name=${exps_name}\
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
                         train.linear_probe=False\
                         train.data_dir="data_ours"

python -m cliport.train  train.task=packing-omni-objects\
                         train.agent=${agent_name}\
                         train.exp_folder=${exps_name}\
                         wandb.run_name=${exps_name}\
                         train.n_demos=1000 \
                         train.n_steps=60100 \
                         train.lr_scheduler=True\
                         train.lr=2e-5\
                         train.warmup_epochs=10\
                         train.precision=32\
                         train.batch_size=1\
                         train.batchnorm=True\
                         train.load_from_last_ckpt=False\
                         train.log=False\
                         mae_model=${mae_model} \
                         pretrain_path=${pretrain_path}\
                         cliport_checkpoint=False\
                         dataset.cache=False \
                         train.sep_mode=place\
                         train.linear_probe=False\
                         train.data_dir="data_ours"

python cliport/eval_sep.py model_task=packing-omni-objects\
                    eval_task=packing-omni-objects-intra\
                    agent=${agent_name} \
                    mode=test \
                    n_demos=100 \
                    train_demos=1000 \
                    exp_folder=${exps_name} \
                    checkpoint_type=best \
                    update_results=True \
                    disp=False\
                    record.save_video=False\
                    data_dir="data_ours"

python cliport/eval_sep.py model_task=packing-omni-objects\
                    eval_task=packing-omni-objects-group-intra \
                    agent=${agent_name} \
                    mode=test \
                    n_demos=100 \
                    train_demos=1000 \
                    exp_folder=${exps_name} \
                    checkpoint_type=best \
                    update_results=True \
                    disp=False\
                    record.save_video=False\
                    data_dir="data_ours"