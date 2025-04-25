# Scripts for mae pre-training (debug mode) 

export CLIPORT_ROOT=$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/mae

export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

python mae/main_pretrain_ours.py \
    --model mae_fuse \
    --batch_size 96 \
    --input_size 224 224 \
    --output_dir  exps/debug \
    --pretrain checkpoints/mae_pretrain_vit_base.pth\
    --mask_ratio 0.95 \
    --epochs 100 \
    --multisize