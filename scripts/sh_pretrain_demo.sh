export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/mae
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

echo "MASTER_ADDR set to $MASTER_ADDR"
echo "MASTER_PORT set to $MASTER_PORT"

# python mae/main_pretrain_ours.py \
#     --model mae_robot_lang \
#     --batch_size 1 \
#     --demo \
#     --resume /jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/output_mae_robot_lang2/checkpoint-160.pth

# python mae/visualization.py \
#     --model mae_robot_lang \
#     --batch_size 1 \
#     --resume /jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/output_mae_robot_lang_big/checkpoint-340.pth

python mae/main_pretrain_ours.py \
    --model mae_robot_lang_noref \
    --batch_size 96 \
    --output_dir debug \
    --pretrain /jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/checkpoints/mae_pretrain_vit_base.pth \