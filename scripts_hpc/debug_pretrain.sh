# Scripts for mae pre-training (debug mode) 

export CLIPORT_ROOT=$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/mae

export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

python mae/main_pretrain_ours.py \
    --model mae_fuse_tt \
    --batch_size 64 \
    --input_size 224 224 \
    --output_dir  exps/debug \
    --pretrain checkpoints/mae_pretrain_vit_base.pth\
    --data_path scratch/bridge_256_train.hdf5 \
    --test_path scratch/bridge_256_val.hdf5\
    --mask_ratio 0.95 \
    --epochs 100 \
