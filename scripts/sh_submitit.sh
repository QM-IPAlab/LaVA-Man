export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/mae

python mae/submitit_pretrain.py \
    --job_dir output \
    --nodes 1 \
    --batch_size 64 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 400 \
    --warmup_epochs 40 \
    --blr 1.5e-4 \
    --weight_decay 0.05