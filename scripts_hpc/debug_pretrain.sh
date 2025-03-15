# Scripts for mae pre-training (debug mode) 

export CLIPORT_ROOT=$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/mae

export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

torchrun --nproc_per_node 1 mae/main_pretrain_ours.py \
    --model mae_fuse \
    --batch_size 80 \
    --input_size 224 224 \
    --output_dir  exps/debug \
    --pretrain False\
    --mask_ratio 0.95 \
    --data_path scratch/bridge_256_train.hdf5 \
    --test_path scratch/bridge_256_val.hdf5\
    --epochs 100 \
