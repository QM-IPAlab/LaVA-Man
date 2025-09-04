#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=42
#SBATCH --mem-per-cpu=3850
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --partition=gpu
#SBATCH --time=10:00:00
#SBATCH --account=su008-acw694
#SBATCH --mail-type=ALL
#SBATCH --mail-user=chaoran.zhu@qmul.ac.uk


export PYTHONPATH=$PYTHONPATH:$(pwd)
module load Miniconda3/4.12.0
source activate qwen

python tools/choose_similar_language.py \
    --hdf5_path scratch/mae-data/ego4d_interactive.hdf5\
    --out_hdf5_path ego4d_open_door.hdf5 \
    --robot_action "open the door"\
    --max_preview 10 \
    --model_name Qwen/Qwen3-4B-Instruct-2507\
    --batch_size 128

#h5repack -v scratch/mae-data/ego4d_interactive.hdf5.backup scratch/mae-data/ego4d_interactive.hdf5.repaired