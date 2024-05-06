# #!/bin/bash
# #SBATCH --partition=small
# #SBATCH --gres=gpu:1
# #SBATCH --job-name=gendemo
# #SBATCH --cpus-per-task=16
# module load python/anaconda3
# source activate mae-cliport
export CLIPORT_ROOT=$(pwd)
python -m cliport.demos_top_down n=1 \
                        task=packing-unseen-google-objects-group \
                        mode=train\
                        save_type=hdf5