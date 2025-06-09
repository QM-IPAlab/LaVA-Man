"""
Construct a HDF5 file from the real image dataset (RT-1-X)
Store the first and the last image of each episode (from two different views), along with the language instruction.

python build_real_hdf5_multi_view.py -s train -o debug

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
import random

os.environ['CURL_CA_BUNDLE'] = '/etc/ssl/certs/ca-bundle.crt'

parser = argparse.ArgumentParser(description="Evaluate validation data.")
parser.add_argument("-s", "--split", type=str, default="train", help="train or val split")
parser.add_argument("-o", "--output_name", type=str, default="real_img", help="output name of the dataset")

args = parser.parse_args()

nltk.download('words')
word_list = set(words.words())

dataset_meta = [
    {'name': 'bridge:1.0.0',
     'img_keys': [['observation', 'image_0'],
                  ['observation', 'image_1'],
                  ['observation', 'image_2'],
                  ['observation', 'image_3']],
     'lang_key': ['language_instruction']
    },
]


data_dir= '/data/home/acw694/CLIPort_new_loss/scratch/tensorflow_datasets'

# Take just take the first and the last step
def episode2steps(episode):
    return episode['steps']

def filter_first_last_terminal(step):
    return step['is_first'] | step['is_last']

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

def get_non_zero_images_with_indices(data, img_keys):
    """
    获取非零图像及其对应索引
    """
    images = []
    indices = []
    for idx, key in enumerate(img_keys):
        img = get_nested_value(data, key)
        if not np.all(img == 0):  # 过滤掉全零图像
            images.append(img)
            indices.append(idx)
    
    return images, indices

def compute_mask(img1, img2):
    diff = cv2.absdiff(img1, img2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    return mask

for data in dataset_meta:
    dataset_name = data['name']
    lang_key = data['lang_key']
    img_keys = data['img_keys']

    data_s1 = []
    data_s2 = []
    data_cv1 = [] # cross-view1
    data_cv2 = [] # cross-view2
    data_language = []
    
    f = h5py.File(os.path.join('/data/home/acw694/CLIPort_new_loss/scratch/data_hdf5',
                               f'{args.output_name}.hdf5'), 'a')
    
    ds, ds_info = tfds.load(
        dataset_name, data_dir=data_dir, download=False, split=args.split, with_info=True)
    print(f"Successfully loaded dataset: {dataset_name}")

    ds_steps = ds.map(episode2steps, num_parallel_calls=tf.data.AUTOTUNE).flat_map(lambda x: x)
    filtered_ds = ds_steps.filter(filter_first_last_terminal)
    filtered_ds = filtered_ds.as_numpy_iterator()

    current_first_instruction = None
    current_first_images = None
    current_first_indices = None
    
    print(f"Start processing dataset: {dataset_name}")
    for n_sample, batch in enumerate(filtered_ds):

        if (n_sample + 1) % 1000 == 0: 
            print(f"processed {n_sample} samples")
            
        if batch['is_first']:
            batch_start = batch
            current_first_instruction = get_nested_value(batch, lang_key)
            current_first_images, current_first_indices = get_non_zero_images_with_indices(batch, img_keys)

        elif current_first_instruction and batch['is_last']:
            last_instruction = get_nested_value(batch, lang_key)
            last_images, last_indices = get_non_zero_images_with_indices(batch, img_keys)

            first_valid = is_valid_language(current_first_instruction)
            last_valid = is_valid_language(last_instruction)

            # only when cross-view images are available
            if len(current_first_images) >= 2 and len(last_images) >= 2 and first_valid and last_valid:
                
                # random choose index of start image
                start_idx = random.choice(current_first_indices)
                start_img = get_nested_value(batch_start, img_keys[start_idx])

                # 选取 End 的 index（必须和 Start 的 index 不同）
                crossview_idx_choices = [idx for idx in last_indices if idx != start_idx]
                if not crossview_idx_choices:
                    warnings.warn("No valid cross-view index available, skipping.")
                    continue

                crossview_idx = random.choice(crossview_idx_choices)
                target_image = get_nested_value(batch, img_keys[start_idx])
                crossview_target_image = get_nested_value(batch, img_keys[crossview_idx])
                crossview_start_image = get_nested_value(batch_start, img_keys[crossview_idx])

                # 存储数据
                data_s1.append(start_img)
                data_s2.append(target_image)
                data_cv1.append(crossview_start_image)
                data_cv2.append(crossview_target_image)
                data_language.append(current_first_instruction)

                batch_start = None
            else:
                warnings.warn("Start/End images are insufficient or instruction invalid, skipping.")

            current_first_instruction = None
            current_first_images = None
            current_first_indices = None

        else:
            warnings.warn("Not the first or last, skipping.")

    data_s1 = np.array(data_s1)
    data_s2 = np.array(data_s2)
    data_cv1 = np.array(data_cv1)
    data_cv2 = np.array(data_cv2)
    data_language = np.array(data_language, dtype=h5py.special_dtype(vlen=bytes))

    n1 = append_or_create_dataset(f, 'image_s1', data=data_s1)
    n2 = append_or_create_dataset(f, 'image_s2', data=data_s2)
    n3 = append_or_create_dataset(f, 'language', data=data_language, dtype=h5py.special_dtype(vlen=bytes))
    n4 = append_or_create_dataset(f, 'image_cv1', data=data_cv1)
    n5 = append_or_create_dataset(f, 'image_cv2', data=data_cv2)
    f.close()

    assert n1 == n2 == n3 == n4 == n5

    print(f'Saved {len(data_s1)} samples to the hdf5 file.')
    print(f'Current number of samples in hdf5 file: {n3}.')
