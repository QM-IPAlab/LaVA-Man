"""
Construct a real HDF5 file from the real dataset (RT-1-X)
"""

#load the dataset
import os
os.environ['CURL_CA_BUNDLE'] = '/etc/ssl/certs/ca-bundle.crt'
import tensorflow as tf 
import tensorflow_datasets as tfds
import tqdm
import h5py
import numpy as np
# load raw dataset --> replace this with tfds.load(<dataset_name>) on your
# local machine!
# dataset_name = 'bc_z'
dataset_name_list = ['fractal20220817_data',
'bridge',
'taco_play',
'jaco_play',
'roboturk',
'viola',
'berkeley_autolab_ur5',
'language_table',
'stanford_hydra_dataset_converted_externally_to_rlds',
'maniskill_dataset_converted_externally_to_rlds',
'ucsd_kitchen_dataset_converted_externally_to_rlds',
'ucsd_pick_and_place_dataset_converted_externally_to_rlds',
'bc_z',
'berkeley_rpt_converted_externally_to_rlds',
'kaist_nonprehensile_converted_externally_to_rlds',
'stanford_mask_vit_converted_externally_to_rlds',
'asu_table_top_converted_externally_to_rlds',
'stanford_robocook_converted_externally_to_rlds',
'iamlab_cmu_pickup_insert_converted_externally_to_rlds',
'berkeley_fanuc_manipulation',
'cmu_food_manipulation',
'cmu_play_fusion',
'berkeley_gnm_recon']

data_dir= '/data/home/acw694/CLIPort_new_loss/scratch/tensorflow_datasets'

# Take just take the first and the last step
def episode2steps(episode):
    return episode['steps']

def filter_first_last_terminal(step):
    return step['is_first'] | step['is_last']

def step_map_fn(step):
    return {
        'observation': {
            'image': step['observation']['image'],
            'language': step['observation']['natural_language_instruction'],
        }   
    }

def append_or_create_dataset(f, name, data, dtype=None):

    if name in f:
        # If dataset already exists, append to it
        dset = f[name]
        dset.resize(dset.shape[0] + len(data), axis=0)
        dset[-len(data):] = data
        n = len(dset)

    else:
        if dtype is None:
            maxshape = (None,) + data[0].shape
            chunks = (1,) + data[0].shape
            f.create_dataset(name, data=data, maxshape=maxshape,
                                chunks=chunks)
        else:
            maxshape = (None,)
            chunks = (1,)  # For variable-length data
            f.create_dataset(name, data=data, maxshape=maxshape,
                                chunks=chunks, dtype=dtype)

        n = len(data)

    return n

for dataset_name in dataset_name_list:
    
    print(f"Loading dataset: {dataset_name}")
    data_s1 = []
    data_s2 = []
    data_language = []
    f = h5py.File(os.path.join('/data/home/acw694/CLIPort_new_loss/scratch/data_hdf5',
                                    'real_img.hdf5'), 'a')
    # load the dataset
    ds, ds_info = tfds.load(
        dataset_name, data_dir = data_dir, download=False, split='train', with_info=True)

    # first map to steps
    ds_steps = ds.map(
        episode2steps, num_parallel_calls=tf.data.AUTOTUNE).flat_map(lambda x: x)

    # then filtered out the first and the last image
    filtered_ds = ds_steps.filter(filter_first_last_terminal)
    
    # obtain the image in each step
    ds_mapped = filtered_ds.map(step_map_fn, num_parallel_calls=tf.data.AUTOTUNE)

    for i, batch in tqdm.tqdm(
        enumerate(ds_mapped.batch(2).as_numpy_iterator())):
        # Collect images and language data
        image_start = batch['observation']['image'][0]
        image_end = batch['observation']['image'][1]
        language_start = batch['observation']['language'][0]
        language_end = batch['observation']['language'][1]
        
        assert image_start.shape == image_end.shape
        assert language_start == language_end, "Language should be the same for the first and last step"
        
        # Add them to our lists
        data_s1.append(image_start)
        data_s2.append(image_end)
        data_language.append(language_start)
        break

    data_language = np.array(data_language, dtype=h5py.special_dtype(vlen=bytes))
    data_s1 = np.array(data_s1)
    data_s2 = np.array(data_s2)

    n1 = append_or_create_dataset(f, 'image_s1', data=data_s1)
    n2 = append_or_create_dataset(f, 'image_s2', data=data_s2)
    n3 = append_or_create_dataset(f, 'language', data=data_language, dtype=h5py.special_dtype(vlen=bytes))
    f.close()

    assert n1 == n2 == n3

    print(f'Saved {len(data_s1)} samples to the hdf5 file.')
    print(f'Current number of samples in hdf5 file: {n3}.')

