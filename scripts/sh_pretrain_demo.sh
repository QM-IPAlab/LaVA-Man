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

# mae_robot_lang 
# python mae/main_pretrain_ours.py \
#     --model mae_robot_lang \
#     --batch_size 64 \
#     --output_dir debug \
#     --pretrain  /jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/checkpoints/clip_vit_base_patch16_converted.pth\
#     --mask_ratio 0.75 \
#     --data_path /jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/data_hdf5/full_color_seen_obj.hdf5 \
#     --epochs 400 \
#     --tokenizer openai/clip-vit-base-patch16 \

# mae_clip
python mae/main_pretrain_ours.py \
    --model mae_clip \
    --batch_size 64 \
    --output_dir debug \
    --pretrain  /jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/checkpoints/clip_vit_base_patch16.pth\
    --mask_ratio 0.95 \
    --data_path /jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/data_hdf5/full_color_seen_obj.hdf5 \
    --epochs 400 \
    --demo \
    #--text_model openai/clip-vit-base-patch16 \
#    #--condition_free

# condition free
# python mae/main_pretrain_ours.py \
#     --model mae_robot_lang_cf \
#     --batch_size 64 \
#     --output_dir debug \
#     --pretrain  /jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/checkpoints/mae_pretrain_vit_base.pth\
#     --mask_ratio 0.75 \
#     --data_path /jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/data_hdf5/full_color_seen_obj.hdf5 \
#     --epochs 400 \
#     --condition_free \
#     --demo \
    #--text_model openai/clip-vit-base-patch16 \
   
# python mae/main_pretrain_ours.py \
#     --model jepa_2loss \
#     --batch_size 64 \
#     --output_dir debug \
#     --pretrain  /jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/checkpoints/mae_pretrain_vit_base.pth\
#     --mask_ratio 0.95 \
#     --data_path /jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/data_hdf5/full_color_seen_obj.hdf5 \
#     --epochs 400 \
#     --demo \

    
