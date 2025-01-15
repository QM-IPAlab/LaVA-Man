"""
Construct a HDF5 file from the real image dataset (RT-1-X)

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

def resize_and_crop_longest_edge_cv2(image, target_height=320, target_width=160):
    
    # 获取输入图像的高度和宽度
    input_height, input_width = image.shape[:2]
    
     # rotate if width > height
    if input_width > input_height:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE) # type: ignore
    input_height, input_width = image.shape[:2]
    
    # 计算目标的宽高比
    target_aspect_ratio = target_width / target_height
    input_aspect_ratio = input_width / input_height

    # 先根据最长边调整大小，保持宽高比不变
    if input_aspect_ratio > target_aspect_ratio:
        # 图像相对较宽，先调整宽度到目标宽度
        new_width = target_width
        new_height = int(input_height * (target_width / input_width))  # 按比例调整高度
    else:
        # 图像相对较高，先调整高度到目标高度
        new_height = target_height
        new_width = int(input_width * (target_height / input_height))  # 按比例调整宽度

    # 使用 OpenCV 的 resize 方法，保持宽高比
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA) # type: ignore

    # 获取调整后图像的高度和宽度
    current_height, current_width = resized_image.shape[:2]
    
    # 居中裁剪到目标大小（如果调整后图像比目标尺寸大）
    if current_height > target_height:
        start_h = (current_height - target_height) // 2
        resized_image = resized_image[start_h:start_h + target_height, :]
    if current_width > target_width:
        start_w = (current_width - target_width) // 2
        resized_image = resized_image[:, start_w:start_w + target_width]

    # 如果裁剪后的尺寸小于目标尺寸，进行填充
    pad_height = max(target_height - resized_image.shape[0], 0)
    pad_width = max(target_width - resized_image.shape[1], 0)

    # 使用 cv2.copyMakeBorder 进行填充
    padded_image = cv2.copyMakeBorder( # type: ignore
        resized_image,
        pad_height // 2, pad_height - pad_height // 2,
        pad_width // 2, pad_width - pad_width // 2,
        cv2.BORDER_CONSTANT, value=[0, 0, 0]  # 填充黑色 # type: ignore
    )

    return padded_image

def resize_and_crop_longest_edge(image, target_height=320, target_width=160):

    input_height, input_width = image.shape[:2]
    
    # rotate if width > height
    if input_width > input_height:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE) # type: ignore
    input_height, input_width = image.shape[:2]
    
    # scale iamge if the height is less than target height
    if input_height < target_height:
        scale_factor = target_height / input_height
        new_width = int(input_width * scale_factor)
        new_height = target_height
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA) # type: ignore

    cropped_image = center_crop(image, target_height, target_width)
    
    return cropped_image 

def center_crop(image, target_height, target_width):
    input_height, input_width = image.shape[:2]
    
    # Calculate the coordinates for cropping
    start_x = (input_width - target_width) // 2
    start_y = (input_height - target_height) // 2
    
    # Ensure the coordinates are within bounds
    start_x = max(start_x, 0)
    start_y = max(start_y, 0)
    
    cropped_image = image[start_y:start_y + target_height, start_x:start_x + target_width]
    
    return cropped_image

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

    data_s1 = []
    data_s2 = []
    data_language = []
    f = h5py.File(os.path.join(f'/data/home/acw694/CLIPort_new_loss/scratch/data_hdf5',
                                    '{args.output_name}.hdf5'), 'a')
    
    # load the dataset
    ds, ds_info = tfds.load(
        dataset_name, data_dir = data_dir, download=False, split=args.split, with_info=True) # type: ignore
    print(f"Successfully load dataset: {dataset_name}")

    # first map to steps
    ds_steps = ds.map( # type: ignore
        episode2steps, num_parallel_calls=tf.data.AUTOTUNE).flat_map(lambda x: x)

    # then filtered out the first and the last image
    filtered_ds = ds_steps.filter(filter_first_last_terminal)
    filtered_ds = filtered_ds.as_numpy_iterator()

    current_first_instruction = None
    current_first_image = None
    
    print(f"Start processing dataset: {dataset_name}")
    for batch in filtered_ds:
        if batch['is_first']:
            current_first_instruction = get_nested_value(batch, lang_key)
            current_first_image = get_nested_value(batch, image_key)

        elif current_first_instruction and batch['is_last']:
            last_instruction = get_nested_value(batch, lang_key)
            last_image = get_nested_value(batch, image_key)

            # safety check
            first_valid = is_valid_language(current_first_instruction)
            last_valid = is_valid_language(last_instruction)
            if current_first_instruction  == last_instruction and first_valid and last_valid:
                #Add them to our lists
                data_s1.append(resize_and_crop_longest_edge_cv2(current_first_image))
                data_s2.append(resize_and_crop_longest_edge_cv2(last_image))
                data_language.append(current_first_instruction)
            else:
                warnings.warn("Instruction is not the same or not valid, skipping.")

            current_first_instruction = None
            current_first_image = None

        else:
            warnings.warn("Not the first or last, skipping.")

    data_s1 = np.array(data_s1)
    data_s2 = np.array(data_s2)
    data_language = np.array(data_language, dtype=h5py.special_dtype(vlen=bytes))

    n1 = append_or_create_dataset(f, 'image_s1', data=data_s1)
    n2 = append_or_create_dataset(f, 'image_s2', data=data_s2)
    n3 = append_or_create_dataset(f, 'language', data=data_language, dtype=h5py.special_dtype(vlen=bytes))
    f.close()

    assert n1 == n2 == n3

    print(f'Saved {len(data_s1)} samples to the hdf5 file.')
    print(f'Current number of samples in hdf5 file: {n3}.')