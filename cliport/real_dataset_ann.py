"""Image dataset."""

from math import e, pi
import os

from kornia import depth
import numpy as np
from sympy import false
from torch.utils.data import Dataset

from cliport import tasks
from cliport.tasks import cameras
from cliport.utils import utils

from torchvision import transforms
import json

from PIL import Image
import cv2

class RealAnnDataset(Dataset):
    """Dataset for loading real images."""

    def __init__(self, task_name="pack_objects", data_type = 'train', augment=false):
        """A simple RGB-D image dataset."""

        print(f"Loading real 207 dataset...") 

        self.augment = augment
        self.path = "/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/real_annotated"

        self.annotation_file = os.path.join(self.path, 'annotations.json')
        self.in_shape = (320, 160, 6)
        self.pix_size = 0.003125

       # cache
        self._cache = []

        # load the annotaions dataset
        with open(self.annotation_file, 'r') as f:
            self.annotations = json.load(f)
        
        # to tensor
        self.transform = transforms.ToTensor()
        
        # load data
        self.preload_data()

    def preload_data(self):
        """Preload all real images into memory."""
        
        for sample_key, annotation in self.annotations.items():

            img_path = os.path.join(self.path, f"{sample_key}.png")
            depth_path = os.path.join(self.path, f"{sample_key}_depth.png")
            
            # Load image
            img = Image.open(img_path)
            depth = Image.open(depth_path).convert('L')
            pick_coords = (annotation["pick_coordinates"][0], annotation["pick_coordinates"][1])
            place_coords = (annotation["place_coordinates"][0], annotation["place_coordinates"][1])

            # radius
            pick_radius = np.array(annotation["pick_radius"])
            place_radius = np.array(annotation["place_radius"])

            img = np.array(img)
            depth = np.array(depth)
            depth = depth/1000.0


            self._cache.append((img,
                                depth, 
                                pick_coords,
                                place_coords, 
                                annotation["instruction"],
                                pick_radius,
                                place_radius))

    def __len__(self):
        return len(self._cache)

    
    def __getitem__(self, idx):
        # Choose random episode.
        
        img, depth, p0, p1, lang_goal, pick_radius, place_radius = self._cache[idx]
        img = img[:, :, :3]
        img = np.concatenate((img,
                              depth[Ellipsis, None],
                              depth[Ellipsis, None],
                              depth[Ellipsis, None]), axis=2)

        p0_theta = 0
        p1_theta = 0

        if self.augment:
            img, _, (p0, p1), perturb_params = utils.perturb(img, [p0, p1])
        
        img = np.float32(img)
        sample = {
            'img': img,
            'p0': p0, 'p0_theta': p0_theta,
            'p1': p1, 'p1_theta': p1_theta,
            'perturb_params': False,
            'lang_goal': lang_goal,
            'pick_radius': pick_radius,
            'place_radius': place_radius
        }

        return sample, sample
    

def draw_circle_on_image(image, pick_coords, pick_radius, save_path="circle.png"):
    # 确保传入的图像是numpy数组
    if isinstance(image, Image.Image):
        img = np.array(image)  # 将PIL图像转换为numpy数组
    elif isinstance(image, np.ndarray):
        img = image
    else:
        raise ValueError("image must be a PIL image or numpy array")
    
    # 确保坐标和半径为整数
    pick_coords = (int(pick_coords[0]), int(pick_coords[1]))
    pick_radius = int(pick_radius)
    
    # 定义颜色和线条厚度
    pick_color = (0, 255, 0)  # 绿色
    thickness = 2  # 圆的线条厚度
    
    # 画圈
    img_with_circle = img.copy()
    img_with_circle = cv2.circle(img_with_circle, (pick_coords[1],pick_coords[0]), pick_radius, pick_color, thickness)
    
    # 保存结果
    img_with_circle = Image.fromarray(img_with_circle)  # 转换回PIL图像格式
    img_with_circle.save(save_path)
    print(f"Image saved to {save_path}")