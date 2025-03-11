#!/bin/bash
#$ -cwd
#$ -j y
#$ -l h_rt=24:0:0
#$ -l h_vmem=7.5G
#$ -l gpu=1
#$ -l gpu_type=ampere
#$ -N pretrain
#$ -m bea
#$ -l rocky
#$ -pe smp 8     
set -e

export CLIPORT_ROOT=$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/mae

module load miniforge
mamba activate mae-cliport

#export MASTER_ADDR=$(hostname)
#export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

python mae/main_pretrain_ours.py \
    --model mae_fuse \
    --batch_size 80 \
    --input_size 224 224 \
    --output_dir  exps/birdge_droid_fuse \
    --pretrain checkpoints/mae_pretrain_vit_base.pth\
    --mask_ratio 0.95 \
    --data_path scratch/bridge_256_train.hdf5 \
    --test_path scratch/bridge_256_val.hdf5\
    --epochs 400 \
    --my_log