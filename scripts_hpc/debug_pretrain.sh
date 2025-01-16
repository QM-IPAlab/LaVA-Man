# Scripts for mae pre-training (debug mode) 

export CLIPORT_ROOT=$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/mae

python mae/main_pretrain_ours.py \
    --model mae_robot_lang \
    --batch_size 128 \
    --input_size 256 256 \
    --output_dir  /data/home/acw694/CLIPort_new_loss/scratch/lava_man/exps_pre/output_mae_resize \
    --pretrain  /data/home/acw694/CLIPort_new_loss/scratch/lava_man/exps_pre/checkpoints/mae_pretrain_vit_base.pth\
    --mask_ratio 0.95 \
    --data_path /data/home/acw694/CLIPort_new_loss/scratch/data_hdf5/bridge_256_train.hdf5 \
    --test_path /data/home/acw694/CLIPort_new_loss/scratch/data_hdf5/bridge_256_val.hdf5\
    --epochs 400 \
    --my_log
