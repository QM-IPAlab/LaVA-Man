#!/bin/bash
#$ -l h_rt=240:0:0
#$ -l h_vmem=7.5G
#$ -pe smp 12
#$ -l gpu=1
#$ -l gpu_type=ampere
#$ -cwd
#$ -j y
#$ -N pretrain
#$ -m bea

set -e

export CLIPORT_ROOT=$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/mae

module load anaconda3
conda activate mae-cliport

python mae/main_pretrain_ours.py \
    --model mae_robot_lang \
    --batch_size 128 \
    --input_size 256 256 \
    --output_dir  exps/output_mae_resize \
    --pretrain  mae_pretrain_vit_base.pth\
    --mask_ratio 0.95 \
    --data_path bridge_256_train.hdf5 \
    --test_path bridge_256_val.hdf5\
    --epochs 400 \
    --my_log