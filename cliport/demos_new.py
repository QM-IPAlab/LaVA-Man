"""Data collection script. Save the top-down view of objects directly to hdf5 file.
update: 2025-03-09
usage: 
    python cliport/demos_new.py n=100 \
                    task=packing-omni-objects \
                    mode=train \
                    data_dir=data_debug\
"""

import os
import hydra
import numpy as np
import random
import h5py
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

from cliport import tasks
from cliport.environments.environment_ours import EnvironmentWhite as Environment


@hydra.main(config_path='./cfg', config_name='data')
def main(cfg):
    # Initialize environment and task.
    env = Environment(
        cfg['assets_root'],
        disp=cfg['disp'],
        shared_memory=cfg['shared_memory'],
        hz=480,
        record_cfg=cfg['record']
    )
    task = tasks.names[cfg['task']]()
    task.mode = cfg['mode']
    record = cfg['record']['save_video']
    save_data = cfg['save_data']

    # Initialize scripted oracle agent and dataset.
    agent = task.oracle(env)
   
    if task.mode == 'train':
        seed = -2 
    elif task.mode == 'val': # NOTE: beware of increasing val set to >100
        seed = -1 
    elif task.mode == 'test':
        seed = -1 + 10000 
    else:
        raise Exception("Invalid mode. Valid options: train, val, test")

    # buffer for hdf5 
    buffer_img_s1 , buffer_img_s2, buffer_lang = [], [], []
    buffer_size = 1000
    saved = 0
    
    # Collect training data from oracle demonstrations.
    with tqdm(total=cfg['n']) as pbar:
        while saved < cfg['n']:
            episode, total_reward = [], 0
            seed += 2

            if saved<2000: saved+=1; pbar.update(1); continue

            # Set seeds.
            np.random.seed(seed)
            random.seed(seed)

            env.set_task(task)
            obs = env.reset()
            if not obs: continue
            info = env.info
            reward = 0

            # Unlikely, but a safety check to prevent leaks.
            #if task.mode == 'val' and seed > (-1 + 10000):
            #    raise Exception("!!! Seeds for val set will overlap with the test set !!!")

            # Rollout expert policy
            for _ in range(task.max_steps):
                try:
                    act = agent.act(obs, info)
                    episode.append((obs, act, reward, info))
                    lang_goal = info['lang_goal']
                    obs, reward, done, info = env.step(act)
                    total_reward += reward
                except Exception as e:
                    print(f'Error: {e}')
                    continue
                #print(f'Total Reward: {total_reward:.3f} | Done: {done} | Goal: {lang_goal}')
                
                # for visualization
                # imgs = obs['color'][0]
                # plt.imshow(imgs)
                # plt.show()
                
                if done:
                    break
            episode.append((obs, None, reward, info))

            # End video recording
            if record:
                env.end_rec()

            # Save demos
            if total_reward > 0.99: 
                for  n_sample in range(len(episode) - 1):
                    
                    sample = episode[n_sample]
                    goal = episode[n_sample + 1]
                    sample_img = sample[0]['color'][0]
                    goal_img = goal[0]['color'][0]
                    language = sample[3]['lang_goal']

                    buffer_img_s1.append(resize(sample_img,160))
                    buffer_img_s2.append(resize(goal_img,160))
                    buffer_lang.append(language)
                    saved += 1
                    pbar.update(1)             

            else:
                #print("demo not finished, skip!")
                pass

            # Save to HDF5 file
            if len(buffer_img_s1) >= buffer_size:
                with h5py.File('top_down_omniobj_white.hdf5', 'a') as f:
                    append_or_create_hdf5(f, 'image_s1', np.array(buffer_img_s1))
                    append_or_create_hdf5(f, 'image_s2', np.array(buffer_img_s2))
                    append_or_create_hdf5(f, 'language', buffer_lang)      

                buffer_img_s1, buffer_img_s2, buffer_lang = [], [], []
                print(f'Saved {saved} samples to the hdf5 file.')

    # Save any remaining samples
    with h5py.File('top_down_omniobj_white.hdf5', 'a') as f:
        
        if buffer_img_s1:    
            append_or_create_hdf5(f, 'image_s1', np.array(buffer_img_s1))
            append_or_create_hdf5(f, 'image_s2', np.array(buffer_img_s2))
            append_or_create_hdf5(f, 'language', buffer_lang)

        n1 = len(f['image_s1']) # type: ignore
        n2 = len(f['image_s2']) # type: ignore
        n3 = len(f['language']) # type: ignore

        print(f'Current number of samples in hdf5 file: {n1}, {n2}, {n3}.')

def append_or_create_hdf5(hdf_file, dataset_name, data, dtype=None):
    """
    Append data to an existing HDF5 dataset or create a new dataset if it doesn't exist.

    Args:
        hdf_file (h5py.File): The open HDF5 file object.
        dataset_name (str): The name of the dataset to append to or create.
        data (np.ndarray or list of str): The data to store. 
            - For images: Should be a numpy array with dtype uint8 and shape (H, W, C) or (B, H, W, C).
            - For variable-length strings: Provide as a list of strings.
        dtype (h5py.Datatype, optional): The datatype for variable-length data like strings.

    Returns:
        int: The new total number of items in the dataset after appending.
    """

    # Check if data is image data or variable-length string data
    if isinstance(data, np.ndarray):
        # Assert image data type and shape
        assert data.dtype == np.uint8, "Input data must be of type uint8 (0-255 pixel values)."
        
        if data.ndim == 3:
            # Single image: (H, W, C) → Convert to batch (1, H, W, C)
            assert data.shape[2] in [1, 3], "Image must have 1 (grayscale) or 3 (RGB) channels."
            data = np.expand_dims(data, axis=0)
        
        elif data.ndim == 4:
            # Batch of images: (B, H, W, C)
            assert data.shape[3] in [1, 3], "Images must have 1 (grayscale) or 3 (RGB) channels."
        
        else:
            raise ValueError("Input data must have shape (H, W, C) or (B, H, W, C).")

        # Create or append to the dataset
        if dataset_name in hdf_file:
            dset = hdf_file[dataset_name]
            dset.resize(dset.shape[0] + data.shape[0], axis=0)
            dset[-data.shape[0]:] = data
        else:
            maxshape = (None,) + data.shape[1:]  # Allow unlimited growth in the batch dimension
            chunks = (1,) + data.shape[1:]       # Store data in chunks for better performance
            hdf_file.create_dataset(dataset_name, data=data, maxshape=maxshape, chunks=chunks)

        return hdf_file[dataset_name].shape[0]

    elif isinstance(data, list) and all(isinstance(item, str) for item in data):
        # Handle variable-length string data
        dt = h5py.special_dtype(vlen=str) if dtype is None else dtype

        if dataset_name in hdf_file:
            dset = hdf_file[dataset_name]
            dset.resize(dset.shape[0] + len(data), axis=0)
            dset[-len(data):] = data
        else:
            maxshape = (None,)
            chunks = (1,)  # Suitable for variable-length data
            hdf_file.create_dataset(dataset_name, data=data, maxshape=maxshape, chunks=chunks, dtype=dt)

        return hdf_file[dataset_name].shape[0]

    else:
        raise TypeError("Unsupported data type. Expected numpy array for images or list of strings for text.")


def resize(img: np.ndarray, size: int) -> np.ndarray:
    """
    Resize an image such that the smaller edge is scaled to 'size' while maintaining aspect ratio.

    Args:
        img (np.ndarray): Input image (H, W, C) or (H, W) in np.uint8 format.
        size (int): The target size for the smaller edge.

    Returns:
        np.ndarray: Resized image.
    """
    h, w = img.shape[:2]
    
    # 计算缩放比例
    if h < w:
        scale = size / h
    else:
        scale = size / w
    
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    # 使用cv2进行缩放
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    rotated_img = cv2.rotate(resized_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    return rotated_img

if __name__ == '__main__':
    main()
