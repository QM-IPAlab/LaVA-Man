##!/bin/bash
##SBATCH --partition=small
##SBATCH --gres=gpu:1
##SBATCH --job-name=rlang
##SBATCH --cpus-per-task=16
#module load python/anaconda3
#source activate mae-cliport
export CLIPORT_ROOT=$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/mae
#export MASTER_ADDR=$(hostname)
#export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

python -m torch.distributed.launch mae/main_pretrain_ours.py \
    --model mae_robot_lang \
    --batch_size 64 \
    --output_dir debug \
    --pretrain /jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/checkpoints/mae_pretrain_vit_base.pth \
    --mask_ratio 0.95 \
    --world_size 4