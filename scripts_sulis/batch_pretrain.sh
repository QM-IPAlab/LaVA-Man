#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=42
#SBATCH --mem-per-cpu=3850
#SBATCH --gres=gpu:ampere_a100:3
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --account=su008-acw694

export CLIPORT_ROOT=$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/mae

module load Miniconda3/4.12.0
source activate mae-cliport

export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

torchrun --nproc_per_node 3 mae/main_pretrain_ours.py \
    --model mae_single \
    --batch_size 96 \
    --input_size 224 224  \
    --output_dir  exps/single_3dataset \
    --mask_ratio 0.75 \
    --data_path bridge_256_train.hdf5 \
    --test_path bridge_256_val.hdf5\
    --epochs 400 \
    --my_log \