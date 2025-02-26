#!/bin/bash
#$ -cwd
#$ -j y
#$ -l h_rt=24:0:0
#$ -l h_vmem=7.5G
#$ -l gpu=2
#$ -l gpu_type=ampere
#$ -N pretrain
#$ -m bea
#$ -l rocky
#$ -pe smp 16     
set -e

export CLIPORT_ROOT=$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/mae

module load miniforge
mamba activate mae-cliport

torchrun --nproc_per_node 2 mae/main_pretrain_ours.py \
    --model mae_single \
    --batch_size 128 \
    --input_size 224 224 \
    --output_dir  exps/single3 \
    --pretrain checkpoints/mae_pretrain_vit_base.pth\
    --mask_ratio 0.75 \
    --data_path scratch/bridge_256_train.hdf5 \
    --test_path scratch/bridge_256_val.hdf5\
    --epochs 100 \
    --condition_free \
    --my_log