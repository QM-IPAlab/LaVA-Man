# #!/bin/bash
# #SBATCH --partition=small
# #SBATCH --gres=gpu:1
# #SBATCH --job-name=gen_data
# #SBATCH --cpus-per-task=16
# #SBATCH --array=0-1
# #SBATCH --time=24:00:00
# module load python/anaconda3
# source activate mae-cliport
export CLIPORT_ROOT=$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/mae

tasks=(
  'assembling-kits-seq-full'
  'packing-boxes-pairs-full'
  'packing-seen-google-objects-seq'
  'packing-unseen-google-objects-seq'
  'packing-seen-google-objects-group'
  'packing-unseen-google-objects-group'
  'put-block-in-bowl-full'
  'stack-block-pyramid-seq-full'
  'separating-piles-full'
  'towers-of-hanoi-seq-full'
)

task_name=${tasks[$SLURM_ARRAY_TASK_ID]}

python cliport/demo_hdf5.py n=10 \
                        task=stack-block-pyramid-seq-unseen-colors \
                        mode=test \
                        hdf5_path=debug2
