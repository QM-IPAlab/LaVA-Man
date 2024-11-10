"""Image dataset."""

from math import pi
import os

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

class RealDataset(Dataset):
    """Dataset for loading real images."""

    def __init__(self, task_name="pack_objects", data_type = 'train', augment=false):
        """A simple RGB-D image dataset."""

        print(f"Loading real new dataset...") 

    
        self.path = "/jmain02/home/J2AD007/txk47/cxz00-txk47/DiffPort/data-real" # FIXME: hardcoded path
        if data_type == 'train':
            self.path = os.path.join(self.path, task_name, data_type)
            self.augment = True
        elif data_type == 'test_seen':
            self.path = os.path.join(self.path, task_name, 'test', 'seen')
            self.augment = False
        elif data_type == 'test_unseen':
            self.path = os.path.join(self.path, task_name, 'test', 'unseen')
            self.augment = False
        elif data_type == 'train_all':
            self.path = "/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/data_real_idiap"
            self.augment = True
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
        
        def process_and_mark_image(image_path, depth_path, x1, y1, x2, y2):
            # 加载图片
            img = Image.open(image_path)
            depth = Image.open(depth_path)
            
            # 原始图片尺寸
            original_width, original_height = img.size
            
            # 裁剪图片，去掉左边和下边的部分
            crop_left = 10  # 从左边开始裁剪的像素
            crop_bottom = 106  # 从底部开始裁剪的像素
            img_cropped = img.crop((crop_left, 0, original_width, original_height - crop_bottom))
            depth_img_cropped = depth.crop((crop_left, 0, original_width, original_height - crop_bottom))
            
            # 缩放图片至320x160
            img_resized = img_cropped.resize((320, 160), Image.Resampling.LANCZOS)
            depth_img_resized = depth_img_cropped.resize((320, 160), Image.Resampling.NEAREST)

            img_rotated = img_resized.rotate(90, expand=True)
            depth_img_rotated = depth_img_resized.rotate(90, expand=True)
            
            scale_x = 320 / 1600
            scale_y = 160 / 800
            rotated_height = img_resized.width  # 旋转后的高度实际上是旋转前的宽度
            
            # 计算旋转后标记点的新坐标
            x1_rotated = int(y1 * scale_y)
            y1_rotated = int(rotated_height - (x1 - crop_left) * scale_x)
            x2_rotated = int(y2 * scale_y)
            y2_rotated = int(rotated_height - (x2 - crop_left) * scale_x)
            
            return img_rotated, depth_img_rotated ,(y1_rotated,x1_rotated), (y2_rotated,x2_rotated)
        
        for sample_key, annotation in self.annotations.items():

            img_path = os.path.join(self.path, f"{sample_key}_rgb.png")
            depth_path = os.path.join(self.path, f"{sample_key}_depth.png")
            # Load image
            processed_img, precessed_depth, pick_coords_rotated, place_coords_rotated = process_and_mark_image(img_path, 
                depth_path,
                annotation["pick_coordinates"][0], annotation["pick_coordinates"][1], 
                annotation["place_coordinates"][0], annotation["place_coordinates"][1]
            )

            # radius
            pick_radius = np.array(annotation["pick_radius"])/5.0
            place_radius = np.array(annotation["place_radius"])/5.0

            img = np.array(processed_img)
            depth = np.array(precessed_depth)
            self._cache.append((img, 
                                depth, 
                                pick_coords_rotated, 
                                place_coords_rotated, 
                                annotation["instruction"],
                                pick_radius,
                                place_radius))
            
            #mask = np.zeros((320, 160))
            #mask = cv2.circle(mask, (pick_coords_rotated[1],pick_coords_rotated[0]), int(pick_radius), 1, -1)
            #mask = cv2.circle(mask, (place_coords_rotated[1],place_coords_rotated[0]), int(place_radius), 1, -1)
            #img = img[:, :, :3]
            #img[mask==1] = [255, 0, 0]
            #cv2.imwrite(f"mask_{sample_key}.png", img)
            #sinput("Press Enter to continue...")


    def __len__(self):
        return len(self._cache)

    
    def __getitem__(self, idx):
        # Choose random episode.
        
        img, depth, p0, p1, lang_goal, pick_radius, place_radius = self._cache[idx]

        img = img[:, :, :3]
        depth = depth[:, :, 0]
        depth = depth / 1000.0

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