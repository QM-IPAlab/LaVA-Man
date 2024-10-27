#!/bin/bash
#SBATCH --partition=big
#SBATCH --gres=gpu:8
#SBATCH --job-name=mae_clip
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=ALL
#SBATCH --mail-user=chaoran.zhu@qmul.ac.uk

# Function to find an idle port around 29500
find_idle_port() {
  # First check port 29500
  PORT=29500
  if ! ss -tuln | grep -q ":$PORT "; then
    echo "Using port $PORT"
    return
  fi

  # If port 29500 is not available, randomly select a port between 29501 and 29599
  while true; do
    PORT=$((29501 + RANDOM % 99))
    if ! ss -tuln | grep -q ":$PORT "; then
      echo "Using port $PORT"
      break
    fi
  done
}

# Find an idle port
find_idle_port
module load python/3.8
module load cuda/12.4
source py-mae-cliport/bin/activate
export CLIPORT_ROOT=$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/mae
export TOKENIZERS_PARALLELISM=false

#export MASTER_ADDR=$(hostname)
#export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

# python -m torch.distributed.launch --nproc_per_node 4 --master_port=$PORT mae/main_pretrain_ours.py \
#     --model mae_robot_lang_rev \
#     --batch_size 64 \
#     --output_dir output_mae_robot_lang_full_color_reverse_aug \
#     --pretrain  /jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/output_mae_robot_lang_full_color_reverse_aug/ck60.pth\
#     --mask_ratio 0.95 \
#     --data_path /jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/data_hdf5/full_color_seen_obj.hdf5 \
#     --epochs 400 \
#     --log \
#     --aug \
#     --transform flip \

# torchrun --nproc_per_node 8 --master_port=$PORT mae/main_pretrain_ours.py \
#     --model jepa_robot_lang \
#     --batch_size 64 \
#     --output_dir output_jepa_extra \
#     --pretrain  /jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/output_jepa_extra/ck120.pth\
#     --mask_ratio 0.95 \
#     --data_path /jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/data_hdf5/extra_dataset_no_aug.hdf5 \
#     --epochs 160 \
#     --my_log

# torchrun --nproc_per_node 8 --master_port=$PORT mae/main_pretrain_ours.py \
#     --model voltron \
#     --batch_size 64 \
#     --output_dir output_voltron_vcond_new \
#     --pretrain /jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/cache/v-cond/v-cond.pt\
#     --mask_ratio 0.75 \
#     --data_path /jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/data_hdf5/extra_full_color_obj.hdf5 \
#     --epochs 400 \
#     --my_log


# torchrun --nproc_per_node 8 --master_port=$PORT mae/main_pretrain_ours.py \
#     --model mae_robot_lang \
#     --batch_size 64 \
#     --output_dir output_extra_clip_mae2 \
#     --pretrain /jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/checkpoints/clip_vit_base_patch16_converted.pth\
#     --mask_ratio 0.95 \
#     --data_path /jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/data_hdf5/extra_full_color_obj.hdf5 \
#     --epochs 400 \
#     --my_log \
#     --text_model openai/clip-vit-base-patch16 \


# torchrun --nproc_per_node 8 --master_port=$PORT mae/main_pretrain_ours.py \
#     --model mae_robot_lang_cf \
#     --batch_size 64 \
#     --output_dir output_extra_cf_deepcopy \
#     --pretrain /jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/checkpoints/mae_pretrain_vit_base.pth\
#     --mask_ratio 0.95 \
#     --data_path /jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/data_hdf5/extra_full_color_obj.hdf5 \
#     --epochs 400 \
#     --my_log \
#     --condition_free

torchrun --nproc_per_node 8 --master_port=$PORT mae/main_pretrain_ours.py \
    --model robot_clip \
    --batch_size 64 \
    --output_dir output_robot_clip_nofreeze_2 \
    --pretrain /jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/checkpoints/mae_pretrain_vit_base.pth\
    --mask_ratio 0.95 \
    --data_path /jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/data_hdf5/extra_dataset_no_aug.hdf5 \
    --epochs 400 \
    --my_log \


# torchrun --nproc_per_node 8 --master_port=$PORT mae/main_pretrain_ours.py \
#     --model mae_clip \
#     --batch_size 64 \
#     --output_dir output_mae_clip_2 \
#     --pretrain /jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/checkpoints/mae_pretrain_vit_base.pth\
#     --mask_ratio 0.95 \
#     --data_path /jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/data_hdf5/extra_dataset_no_aug.hdf5 \
#     --epochs 400 \
#     --my_log \