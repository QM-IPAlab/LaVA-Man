#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=3850
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --account=su008-acw694

export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/mae

module load Miniconda3/4.12.0
source activate mae-cliport


# Train voltron ours
# python mae/eval_refer.py \
#     --model voltron \
#     --pretrain  /home/a/acw694/CLIPort_new_loss/checkpoints/voltron_ours_pretrain.ckpt\
#     --mask_ratio 0.75  \
#     --text_model distilbert-base-uncased

#Train mae_robot_lang
python mae/eval_refer.py \
    --model mae_robot_lang \
    --pretrain  /home/a/acw694/CLIPort_new_loss/checkpoints/mae_robot_lang_mix_160.pth\
    --mask_ratio 0.95  \