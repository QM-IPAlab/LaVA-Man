import tensorflow_datasets as tfds
import tqdm

# optionally replace the DATASET_NAMES below with the list of filtered datasets from the google sheet
DATASET_NAMES = ['fractal20220817_data', 'bridge', 'taco_play', 'jaco_play', 'roboturk', 'viola', 'berkeley_autolab_ur5', 'language_table', 'stanford_hydra_dataset_converted_externally_to_rlds', 'maniskill_dataset_converted_externally_to_rlds', 'ucsd_kitchen_dataset_converted_externally_to_rlds', 'ucsd_pick_and_place_dataset_converted_externally_to_rlds', 'bc_z', 'berkeley_rpt_converted_externally_to_rlds', 'kaist_nonprehensile_converted_externally_to_rlds', 'stanford_mask_vit_converted_externally_to_rlds',   'asu_table_top_converted_externally_to_rlds', 'stanford_robocook_converted_externally_to_rlds', 'iamlab_cmu_pickup_insert_converted_externally_to_rlds',  'berkeley_fanuc_manipulation', 'cmu_food_manipulation', 'cmu_play_fusion', 'berkeley_gnm_recon']
DOWNLOAD_DIR = 'scratch/tensorflow_datasets'

print(f"Downloading {len(DATASET_NAMES)} datasets to {DOWNLOAD_DIR}.")
for dataset_name in tqdm.tqdm(DATASET_NAMES):
  _ = tfds.load(dataset_name, data_dir=DOWNLOAD_DIR)
