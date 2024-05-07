"""
Data collection script.
Save the top-down view of ojbects for training the diffusion model.
"""
import torch
import os
import hydra
import numpy as np
import random
import csv
from cliport import tasks
from cliport.dataset import RavensDataset
from cliport.environments.environment_top_down import EnvironmentTopDown as Environment
from skimage.transform import resize
from cliport.utils.utils import xyz_to_pix,quatXYZW_to_eulerXYZ

import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

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
    save_type = cfg['save_type']

    # Initialize scripted oracle agent and dataset.
    agent = task.oracle(env)
    data_path = os.path.join(cfg['data_dir'], "{}-{}".format(cfg['task'], task.mode))
    dataset = RavensDataset(data_path, cfg, n_demos=0, augment=False)

    # Train seeds are even and val/test seeds are odd. Test seeds are offset by 10000
    seed = dataset.max_seed
    if seed < 0:
        if task.mode == 'train':
            seed = -2
        elif task.mode == 'val': # NOTE: beware of increasing val set to >100
            seed = -1
        elif task.mode == 'test':
            seed = -1 + 10000
        else:
            raise Exception("Invalid mode. Valid options: train, val, test")
 
    if save_type == 'hdf5':
        print('Data will be saved to: data_hdf5')
        f = h5py.File(os.path.join('data_hdf5', 'images.hdf5'), 'w')
        folder_name = 'data_hdf5'
    else:
        print('Data will be saved to: data_image')
        csv_data = []
        folder_name = 'data_image'
    
    data_s1 = [] #TODO： dynamic save image data ?
    data_s2 = []
    data_language = []
    data_gt_pick = []
    data_gt_place = []
    
    # parameters for converting xyz to pixel
    bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])
    pix_size = 0.003125

    # collect data
    for i in tqdm(range(int(cfg['n']))):
        episode, total_reward = [], 0
        seed += 2

        # Set seeds.
        np.random.seed(seed)
        random.seed(seed)

        # Set task.
        env.set_task(task)
        obs = env.reset()
        info = env.info
        # Just first image of each tasks
        # image = obs['color'][0]
        # image = resize(image, (160, 320))
        # data_image.append(image)

        # language_goal = info['lang_goal']
        # data_language.append(language_goal)

        # act = agent.act(obs, info)
        # p0_xyz, p0_xyzw = act['pose0']
        # p1_xyz, p1_xyzw = act['pose1']
        # p0 = xyz_to_pix(p0_xyz, bounds, pix_size)
        # p0_theta = -np.float32(quatXYZW_to_eulerXYZ(p0_xyzw)[2])
        # p1 = xyz_to_pix(p1_xyz, bounds, pix_size)
        # p1_theta = -np.float32(quatXYZW_to_eulerXYZ(p1_xyzw)[2])
        # data_gt_pick.append((p0, p0_theta))
        # data_gt_place.append((p1, p1_theta))

        #Rollout expert policy ： This will generate images within the same tasks       
        for _ in range(task.max_steps):
            
            act = agent.act(obs, info)
            lang_goal = info['lang_goal']

            # If the next state exists
            # get the location of pick and place
            if act is not None and lang_goal is not None:
                
                # gt location for pick and place
                p0_xyz, p0_xyzw = act['pose0']
                p1_xyz, p1_xyzw = act['pose1']
                p0 = xyz_to_pix(p0_xyz, bounds, pix_size)
                p0_theta = -np.float32(quatXYZW_to_eulerXYZ(p0_xyzw)[2])
                p1 = xyz_to_pix(p1_xyz, bounds, pix_size)
                p1_theta = -np.float32(quatXYZW_to_eulerXYZ(p1_xyzw)[2])
                
                data_gt_pick.append(np.array([p0[0],p0[1],p0_theta]))
                data_gt_place.append(np.array([p1[0],p1[1],p1_theta]))

                # observation
                image_s1 = obs['color'][0]
                image_s1 = resize(image_s1, (160, 320))
                data_s1.append(image_s1)

                # language goal
                data_language.append(lang_goal)
            
            else:
                break        

            # next step
            obs, reward, done, info = env.step(act)
            
            image_s2 = obs['color'][0]
            image_s2 = resize(image_s2, (160, 320))
            data_s2.append(image_s2)
            
            if done:
                break

    if save_type == 'hdf5':

        language_encoded = np.array(data_language, dtype=h5py.special_dtype(vlen=str))

        f.create_dataset('image_s1', data=data_s1)
        f.create_dataset('image_s2', data=data_s2)
        f.create_dataset('language', data=language_encoded, dtype=h5py.string_dtype(encoding='ascii'))
        f.create_dataset('gt_pick', data=data_gt_pick)
        f.create_dataset('gt_place', data=data_gt_place)
        f.close()
    
    #Save top-down view of the scene to the image
    else:
                
        csv_file_name = f'{folder_name}/metadata.csv'
        print('file_name,text',file=open(csv_file_name, 'w'))
        
        # save images
        for n in range(len(data_s1)):
            csv_data.append([f'{folder_name},{data_language[n]}']) 
            # sample name
            name = f'{n:04d}'
            
            # save s1 image
            image_s1 = data_s1[n]
            image_s1 = convert_to_opencv_format(image_s1)
            image_s1 = draw_circle_on_image(image_s1, data_gt_pick[n][0], data_gt_pick[n][1])
            cv2.imwrite(f'{folder_name}/{name}_s1.png', image_s1)

            # save s2 image
            image_s2 = data_s2[n]
            image_s2 = convert_to_opencv_format(image_s2)
            cv2.imwrite(f'{folder_name}/{name}_s2.png', image_s2)

        with open(csv_file_name, 'a') as f:
                for row in csv_data:
                    print(*row,file=f)
               

def convert_to_opencv_format(image):
    # 检查输入图像数据类型并相应转换
    if image.dtype == np.float32 or image.dtype == np.float64:
        # 假设像素值范围为0.0到1.0，将其转换为0到255的整数
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            # 如果像素值已经大于1，直接转换为uint8
            image = image.astype(np.uint8)
    elif image.dtype != np.uint8:
        # 如果数据类型不是uint8，也转换为uint8
        image = image.astype(np.uint8)
    
    if image.ndim == 3 and image.shape[2] == 3:  # 确保它是一个具有三个通道的图像
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

def draw_circle_on_image(image, x, y, r=5):
    cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 1)
    return image


if __name__ == '__main__':
    main()
