#!/bin/bash
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --job-name=gen_data
#SBATCH --cpus-per-task=16

export CUDA_VISIBLE_DEVICES=1
export CLIPORT_ROOT=$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/mae

# tasks=(
#   'assembling-kits-seq-full'
#   'packing-boxes-pairs-full'
#   'put-block-in-bowl-full'
#   'stack-block-pyramid-seq-full'
#   'separating-piles-full'
#   'towers-of-hanoi-seq-full'
#   'packing-shapes'
#   'align-rope'
# )


# for task in "${tasks[@]}"
# do
#     echo "Running data colloection for task: $task"
#     python cliport/demos.py n=100 \
#                             task=$task \
#                             mode=val \
#                             data_dir=data\
#                             disp=False
# done

# for task in "${tasks[@]}"
# do
#     echo "Running data colloection for task: $task"
#     python cliport/demos.py n=100 \
#                             task=$task \
#                             mode=test \
#                             data_dir=data\=
#                             disp=False
# done

# python cliport/demos_ours.py n=1000 \
#                         task=packing-omni-objects \
#                         mode=train \
#                         data_dir=data_ours\
#                         record.save_video=False \
#                         disp=False \

# python cliport/demos_ours.py n=100 \
#                         task=packing-omni-objects \
#                         mode=val \
#                         data_dir=data_ours\
#                         record.save_video=False \
#                         disp=False \

# python cliport/demos_ours.py n=100 \
#                         task=packing-omni-objects-intra \
#                         mode=test \
#                         data_dir=data_ours\
#                         record.save_video=False \
#                         disp=False \

python cliport/demos_ours.py n=100 \
                        task=packing-omni-objects-group \
                        mode=test \
                        data_dir=data_ours\
                        record.save_video=False \
                        disp=False \

python cliport/demos_ours.py n=100 \
                        task=packing-omni-objects-group-inter \
                        mode=test \
                        data_dir=data_ours\
                        record.save_video=False \
                        disp=False \

python cliport/demos_ours.py n=100 \
                        task=packing-omni-objects-group-intra \
                        mode=test \
                        data_dir=data_ours\
                        record.save_video=False \
                        disp=False \
