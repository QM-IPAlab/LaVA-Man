#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=big
#SBATCH --gres=gpu:4
#SBATCH --job-name=extra
#SBATCH --cpus-per-task=16

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
module load python/anaconda3
source activate mae-cliport
export CLIPORT_ROOT=$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/mae
export OMP_NUM_THREADS=1
#export MASTER_ADDR=$(hostname)
#export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

python -m torch.distributed.launch --nproc_per_node 4 --master_port=$PORT mae/main_pretrain_ours.py \
    --model mae_robot_lang \
    --batch_size 64 \
    --output_dir output_mae_robot_lang_big_extra \
    --pretrain /jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/output_mae_robot_lang_big/checkpoint-160.pth \
    --mask_ratio 0.95 \
    --log \
    --data_path /jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/data_hdf5/extra_dataset_no_aug.hdf5
    #--wandb_resume zqv1t9oz