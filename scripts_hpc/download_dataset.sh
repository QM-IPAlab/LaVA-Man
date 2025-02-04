#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 4       # 4 cores (12 cores per GPU)
#$ -l h_rt=10:0:0    # 10 hour runtime (required to run on the short queue)
#$ -l h_vmem=1G   # 7.5 * 12 = 90G total RAM

module load anaconda3/2023.03
conda activate py3.10

# #DATASET_NAMES='fractal20220817_data bridge taco_play jaco_play roboturk viola berkeley_autolab_ur5 language_table stanford_hydra_dataset_converted_externally_to_rlds maniskill_dataset_converted_externally_to_rlds ucsd_kitchen_dataset_converted_externally_to_rlds ucsd_pick_and_place_dataset_converted_externally_to_rlds bc_z berkeley_rpt_converted_externally_to_rlds kaist_nonprehensile_converted_externally_to_rlds stanford_mask_vit_converted_externally_to_rlds asu_table_top_converted_externally_to_rlds stanford_robocook_converted_externally_to_rlds iamlab_cmu_pickup_insert_converted_externally_to_rlds berkeley_fanuc_manipulation cmu_food_manipulation cmu_play_fusion berkeley_gnm_recon'
# DATASET_NAMES='bridge'
# for name in $DATASET_NAMES
#     do
#         echo "Downing dataset...: $name"
#         gsutil -m cp -r gs://gresearch/robotics/$name /data/home/acw694/CLIPort_new_loss/scratch/tensorflow_datasets/
#     done
# echo "Finished Tasks."

wget -r -np -nH --cut-dirs=4 -R "index.html*" -P /data/home/acw694/CLIPort_new_loss/scratch/tmp_bridge https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/
