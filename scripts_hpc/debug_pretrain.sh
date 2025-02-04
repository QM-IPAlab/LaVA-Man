# Scripts for mae pre-training (debug mode) 

export CLIPORT_ROOT=$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/mae

export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

python mae/main_pretrain_ours.py \
    --model mae_robot_lang \
    --batch_size 96 \
    --input_size 256 256 \
    --output_dir  exps/output_mae_resize \
    --pretrain  mae_pretrain_vit_base.pth\
    --mask_ratio 0.95 \
    --data_path bridge_256_train.hdf5 \
    --test_path bridge_256_val.hdf5\
    --epochs 400 \
    --my_log
