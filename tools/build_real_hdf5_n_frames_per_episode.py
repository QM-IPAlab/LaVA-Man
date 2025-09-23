"""
Construct a HDF5 file from the real image dataset (RT-1-X)
Store the first and the last image of each episode, along with the language instruction.

python build_real_hdf5_2.py -s train -o name

"""
import os
import tensorflow as tf 
import tensorflow_datasets as tfds
import h5py
import numpy as np
import cv2
import warnings
import nltk
from nltk.corpus import words
import argparse

os.environ['CURL_CA_BUNDLE'] = '/etc/ssl/certs/ca-bundle.crt'

parser = argparse.ArgumentParser(description="Evaluate validation data.")
parser.add_argument("-s", "--split", type=str, default="train", help="train or val split")
parser.add_argument("-o", "--output_name", type=str, default="real_img", help="output name of the dataset")

args = parser.parse_args()

nltk.download('words')
word_list = set(words.words())

dataset_meta = [
    {'name':'bridge:1.0.0',
    'img_key': ['observation','image_0'],
    'lang_key': ['language_instruction']
    },
#    {'name' :'cmu_play_fusion',
#     'img_key': ['observation','image'],
#     'lang_key': ['language_instruction']
#     },
#     {'name' :'jaco_play',
#     'img_key': ['observation','image'],
#     'lang_key': ['observation','natural_language_instruction']
#     }
]

data_dir= 'scratch/tensorflow_datasets'

# Take just take the first and the last step
def episode2steps(episode):
    return episode['steps']

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

def get_nested_value(data, keys):
    value = data
    for key in keys:
        value = value[key]  # 一层一层向下访问
    return value

def is_valid_language(input_data):
    
    # not empty
    if not input_data:
        return False
    
    # can be encoded as ascii
    try:
        input_string = input_data.decode('ascii')
    except UnicodeDecodeError:
        return False
    
    # all words in the input in the word list
    words_in_input = input_string.split()
    for word in words_in_input:
        cleaned_word = word.strip(',.!?')
        if cleaned_word.lower() not in word_list:
            return False

    return True

for data in dataset_meta:

    dataset_name = data['name']
    lang_key = data['lang_key']
    image_key = data['img_key']

    data_img = []
    data_lang = []
    f = h5py.File(os.path.join('/data/home/acw694/CLIPort_new_loss/scratch/data_hdf5',
                                    f'{args.output_name}.hdf5'), 'a')
    
    def _flush_to_hdf5():
        
        if not data_img:
            return
    
        _data_img = np.array(data_img)
        _data_lang = np.array(data_lang, dtype=h5py.special_dtype(vlen=bytes))

        n1 = append_or_create_dataset(f, 'image', data=_data_img)
        n2 = append_or_create_dataset(f, 'language', data=_data_lang, dtype=h5py.special_dtype(vlen=bytes))

        assert n1 == n2
        print(f'Saved {len(data_img)} samples to the hdf5 file.')
        print(f'Current number of samples in hdf5 file: {n2}.')
    
        # “原地”清空列表，外层 data_img/data_lang 也会被清
        data_img.clear()
        data_lang.clear()


    # load the dataset
    ds, ds_info = tfds.load(
        dataset_name, data_dir = data_dir, download=False, split=args.split, with_info=True) # type: ignore
    print(f"Successfully load dataset: {dataset_name}")

    # first map to steps
    ds_steps = ds.map( # type: ignore
        episode2steps, num_parallel_calls=tf.data.AUTOTUNE).flat_map(lambda x: x)

    # then filtered out the first and the last image
    episodes = ds_steps.as_numpy_iterator()
    
    print(f"Start processing dataset: {dataset_name}")
    for batch in episodes:
        if batch['is_first']:
            trajectory = [batch]

        elif batch['is_last']:
            trajectory.append(batch)

            n_steps = len(trajectory)

            if n_steps >= 8:
                indices = np.linspace(0, n_steps - 1, 8, dtype=int)
            else:
                trajectory = []
                continue

            sampled_imgs = []
            sampled_langs = []    
            
            for idx in indices:
                step = trajectory[idx]
                instruction = get_nested_value(step, lang_key)
                image = get_nested_value(step, image_key)
                
                sampled_imgs.append(image)
                sampled_langs.append(instruction)

            first_instruction = sampled_langs[0]
            all_same = all(instruction == first_instruction for instruction in sampled_langs)
            valid = is_valid_language(first_instruction)

            if all_same and valid:
                data_img.extend(sampled_imgs)
                data_lang.extend(sampled_langs)
            else:
                warnings.warn("Instructions are not the same or not valid, skipping.")
        
            trajectory = []

        else:
            trajectory.append(batch)

        if len(data_img) >= 1000:
            _flush_to_hdf5()
        
    # remaining samples
    _flush_to_hdf5()
    f.close()
    print(f"Finished processing dataset: {dataset_name}")
    

