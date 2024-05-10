# module load python/anaconda3
# source activate mae-cliport
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

echo "MASTER_ADDR set to $MASTER_ADDR"
echo "MASTER_PORT set to $MASTER_PORT"

python mae/main_pretrain_ours.py \
    --batch_size 16 \
    --pretrain /jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/checkpoints/mae_pretrain_vit_base.pth
