"""Generate more data to hdf5 file for mae pretraining."""

import os
import hydra
import numpy as np
import random
import h5py
from cliport import tasks
from cliport.dataset_to_hdf5_test import RavensDatasetToHdf5
from cliport.environments.environment import Environment

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
    hdf5_name = cfg['hdf5_path']

    # Initialize scripted oracle agent and dataset.
    agent = task.oracle(env)
    data_path = 'debug'
    dataset = RavensDatasetToHdf5(data_path, cfg, n_demos=0, augment=False)

    # Train seeds are even and val/test seeds are odd. Test seeds are offset by 10000
    seed = dataset.max_seed
    seed = -2 + 100000 # new dataset
    file_path = os.path.join('/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/data_hdf5',
                            f'{hdf5_name}.hdf5')
    f = h5py.File(file_path, 'a')
    print(f'saving to : {file_path}')
    
    # Collect training data from oracle demonstrations.
    data_s1 = []  #TODO： dynamic save image data ?
    data_s2 = []
    data_language = []
    data_gt_pick = []
    data_gt_place = []

    while dataset.n_episodes < cfg['n']:
        
        episode, total_reward = [], 0
        seed += 2

        # Set seeds.
        np.random.seed(seed)
        random.seed(seed)

        print('Oracle demo: {}/{} | Seed: {}'.format(dataset.n_episodes + 1, cfg['n'], seed))

        env.set_task(task)
        obs = env.reset()
        info = env.info
        reward = 0

        # Start video recording (NOTE: super slow)
        if record:
            env.start_rec(f'{dataset.n_episodes+1:06d}')

        # Rollout expert policy
        for _ in range(task.max_steps):
            act = agent.act(obs, info)
            episode.append((obs, act, reward, info))
            lang_goal = info['lang_goal']
            obs, reward, done, info = env.step(act)
            total_reward += reward
            print(f'Total Reward: {total_reward:.3f} | Done: {done} | Goal: {lang_goal}')
            if reward == 0.0:
                print("fail, break")
                break
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
                sample = dataset.process_sample(sample, augment=False)
                goal = dataset.process_goal(goal, perturb_params=False)

                image_s1, image_s2, language_encoded, gt_pick, gt_place = dataset.interpret(sample, goal)
                data_s1.append(image_s1)
                data_s2.append(image_s2)
                data_language.append(language_encoded)
                data_gt_pick.append(gt_pick)
                data_gt_place.append(gt_place)
            dataset.n_episodes+=1
        else:
            print("demo not finished, skip!")

    #turn to format suitable for hdf5
    data_language = np.array(data_language, dtype=h5py.special_dtype(vlen=str))
    data_s1 = np.array(data_s1)
    data_s2 = np.array(data_s2)
    data_gt_pick = np.array(data_gt_pick)
    data_gt_place = np.array(data_gt_place)

    n1 = dataset.append_or_create_dataset(f, 'image_s1', data=data_s1)
    n2 = dataset.append_or_create_dataset(f, 'image_s2', data=data_s2)
    n3 = dataset.append_or_create_dataset(f, 'language', data=data_language, dtype=h5py.string_dtype(encoding='ascii'))
    n4 = dataset.append_or_create_dataset(f, 'gt_pick', data=data_gt_pick)
    n5 = dataset.append_or_create_dataset(f, 'gt_place', data=data_gt_place, )
    f.close()

    assert n1 == n2 == n3 == n4 == n5

    print(f'Saved {len(data_s1)} samples to the hdf5 file.')
    print(f'Current number of samples in hdf5 file: {n5}.')

if __name__ == '__main__':
    main()
