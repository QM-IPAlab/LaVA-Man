import os
import json
import random
from PIL import Image
import h5py
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from transformers import BlipProcessor, BlipForConditionalGeneration

import h5py
import numpy as np

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



# 如果需要使用标准的变换
default_transform = transforms.Compose([
    transforms.Resize(256),          # 先调整大小，确保最小边大于224
    transforms.CenterCrop(224),      # 中心裁剪到224x224
    transforms.ToTensor(),           # 转换为Tensor
])

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to('cuda' if torch.cuda.is_available() else 'cpu') # type: ignore


class ImagePairDataset(Dataset):
    def __init__(self, root_dir, json_file, transform=None, frame_offset=30, num_images=10):
        """
        初始化数据集

        Args:
            root_dir (str): 数据集根目录
            json_file (str): 选择的序列JSON文件
            transform (callable, optional): 应用于图像的变换
            frame_offset (int): 生成图像对时的帧偏移量
            num_images (int): 每个子文件夹随机选择的图片数量
        """
        self.root_dir = root_dir
        self.frame_offset = frame_offset
        self.num_images = num_images

        # 加载JSON文件
        with open(json_file, 'r') as f:
            self.data = json.load(f)

        # 准备图像对列表
        self.image_pairs = self._prepare_image_pairs()

    def _prepare_image_pairs(self):
        image_pairs = []

        for category, folders in self.data.items():  # return key-value
            for folder, frames in folders.items():
                # 如果帧数量少于需要的随机数量，选择全部，否则随机选择指定数量
                if len(frames) < self.num_images:
                    selected_frames = frames
                else:
                    selected_frames = random.sample(frames, self.num_images)

                # 为每个选择的帧生成图像对
                for frame in selected_frames:
                    pair_frame = self._get_pair_frame(frame, frames)
                    if pair_frame is not None:
                        img1_path = os.path.join(self.root_dir, category, folder, 'images', f'frame{frame:06d}.jpg')
                        img2_path = os.path.join(self.root_dir, category, folder, 'images', f'frame{pair_frame:06d}.jpg')
                        image_pairs.append((img1_path, img2_path))

        return image_pairs

    def _get_pair_frame(self, frame, frames):
        """根据给定帧生成±30范围内的帧 确保不越界。"""
        valid_frames = [f for f in frames if abs(f - frame) <= self.frame_offset and f != frame]
        if valid_frames:
            return random.choice(valid_frames)
        return None

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        img1_path, img2_path = self.image_pairs[idx]

        img1_pil = Image.open(img1_path).convert('RGB')
        img2_pil = Image.open(img2_path).convert('RGB')

        img1_np = np.array(img1_pil, dtype=np.uint8)
        img2_np = np.array(img2_pil, dtype=np.uint8)

        return img1_np, img2_np


def generate_caption(image):
    inputs = processor(images=image, return_tensors="pt").to('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        output = model.generate(**inputs) # type: ignore
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption


if __name__ == "__main__":
    # 示例用法
    dataset = ImagePairDataset(
        root_dir='scratch/co3d',  # 数据集根目录
        json_file='scratch/co3d/selected_seqs_train.json',  # JSON文件路径
        transform=default_transform  # 图像转换
    )

    # 创建 HDF5 文件并保存数据
    with h5py.File('image_pairs_with_captions.hdf5', 'w') as hdf:
        captions = []
        
        for idx, (img1_np, img2_np) in enumerate(dataset): # type: ignore
            # 生成文字描述
            caption = generate_caption(img1_np)
            captions.append(caption.encode('ascii'))
            
            append_or_create_hdf5(hdf, 'image_s1', img1_np)
            append_or_create_hdf5(hdf, 'image_s2', img2_np)

            if idx % 10 == 0:
                print(f'Saved {idx + 1} image pairs with captions')

            if idx % 100 == 0: break
        
        # Save all captions at once
        append_or_create_hdf5(hdf, 'language', captions)

    print(f'Saved {len(dataset)} image pairs with captions to image_pairs_with_captions.h5')