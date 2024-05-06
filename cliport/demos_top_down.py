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

    n = 0    
    if save_type == 'hdf5':
        print('Data will be saved to: data_hdf5')
        f = h5py.File(os.path.join('data_hdf5', 'images.hdf5'), 'a')
        folder_name = 'data_hdf5'
    else:
        print('Data will be saved to: data_image')
        csv_data = []
        folder_name = 'data_image'
    
    data_image = [] #TODO： dynamic save image data ?
    data_language = []
    data_gt_pick = []
    data_gt_place = []
    
    # parameters for converting xyz to pixel
    bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])
    pix_size = 0.003125

    print('Data will be saved to: {}'.format(folder_name))

    # collect data
    for i in tqdm(range(int(cfg['n']))):
        episode, total_reward = [], 0
        seed += 2

        # Set seeds.
        np.random.seed(seed)
        random.seed(seed)

        env.set_task(task)
        obs = env.reset()
        info = env.info
        
        image = obs['color'][0]
        image = resize(image, (160, 320))
        data_image.append(image)

        language_goal = info['lang_goal']
        data_language.append(language_goal)

        act = agent.act(obs, info)
        p0_xyz, p0_xyzw = act['pose0']
        p1_xyz, p1_xyzw = act['pose1']
        p0 = xyz_to_pix(p0_xyz, bounds, pix_size)
        p0_theta = -np.float32(quatXYZW_to_eulerXYZ(p0_xyzw)[2])
        p1 = xyz_to_pix(p1_xyz, bounds, pix_size)
        p1_theta = -np.float32(quatXYZW_to_eulerXYZ(p1_xyzw)[2])
        data_gt_pick.append((p0, p0_theta))
        data_gt_place.append((p1, p1_theta))

        # Rollout expert policy ： This will generate images within the same tasks       
        # for _ in range(task.max_steps):
            
        #     act = agent.act(obs, info)
        #     lang_goal = info['lang_goal']

        #     if act is not None and lang_goal is not None:
                
        #         # gt location for pick and place
        #         p0_xyz, p0_xyzw = act['pose0']
        #         p1_xyz, p1_xyzw = act['pose1']
        #         p0 = xyz_to_pix(p0_xyz, bounds, pix_size)
        #         p0_theta = -np.float32(quatXYZW_to_eulerXYZ(p0_xyzw)[2])
        #         p1 = xyz_to_pix(p1_xyz, bounds, pix_size)
        #         p1_theta = -np.float32(quatXYZW_to_eulerXYZ(p1_xyzw)[2])
                
        #         data_gt_pick.append((p0, p0_theta))
        #         data_gt_place.append((p1, p1_theta))

        #         # observation
        #         image = obs['color'][0]
        #         image = resize(image, (160, 320))
        #         data_image.append(image)

        #         # language goal
        #         language_goal = info['lang_goal']
        #         data_language.append(language_goal)

        #     # next step
        #     obs, reward, done, info = env.step(act)
        #     if done:
        #         break

    if save_type == 'hdf5':

        language_encoded = np.array(data_language, dtype=h5py.special_dtype(vlen=str))

        f.create_dataset('images', data=data_image)
        f.create_dataset('language', data=language_encoded, dtype=h5py.string_dtype(encoding='ascii'))
        f.create_dataset('gt_pick', data=data_gt_pick)
        f.create_dataset('gt_place', data=data_gt_place)
        f.close()
    #Save top-down view of the scene to the image
    else:
        for n in range(len(data_image)):
            # name = f'{n:04d}.png'
            # plt.imsave(f'{folder_name}/{name}', data_image[n])
            # csv_data.append([f'{name},{data_language[n]}'])    
            # csv_file_name = f'{folder_name}/metadata.csv'
            # print('file_name,text',file=open(csv_file_name, 'w'))
            # with open(csv_file_name, 'a') as f:
            #     for row in csv_data:
            #         print(*row,file=f)
            # print(f'CSV file "{csv_file_name}" has been created successfully.')
            import matplotlib.patches as patches
            
            fig, ax = plt.subplots()
            ax.imshow(data_image[n])
            
            # 添加 pick 圆圈
            pick_circle = plt.Circle(data_gt_pick[n][0], radius=10, color='red', fill=False)
            ax.add_artist(pick_circle)
            
            # 添加 place 圆圈
            place_circle = patches.Circle(data_gt_place[n][0], radius=10, color='red', fill=False)
            ax.add_artist(place_circle)
            
            # 保存图像
            name = f'{n:04d}.png'
            fig.savefig(f'{folder_name}/{name}')
            plt.close(fig)  # 关闭图形，防止资源泄露



def convert_data(f,split):
   """
   convert data to hdf5 format
   """ 


if __name__ == '__main__':
    main()
