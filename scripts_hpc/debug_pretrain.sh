# Scripts for mae pre-training (debug mode) 

export CLIPORT_ROOT=$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/mae

export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

python mae/main_pretrain_ours.py \
    --model mpi \
    --batch_size 64 \
    --input_size 224 224 \
    --output_dir  exps/debug \
    --mask_ratio 0.75 \
    --epochs 100 \
    #--multisize