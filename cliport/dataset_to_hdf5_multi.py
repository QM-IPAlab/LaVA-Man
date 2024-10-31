import h5py
import numpy as np
import os
import hydra
from cliport.dataset import RavensDataset, RavensMultiTaskDataset


def interpret(sample, goal=None):
        # Interpret the sample and goal data
        image_s1 = sample['img']
        image_s2 = goal['img'] if goal is not None else None
        language = sample['lang_goal']
        gt_pick = [sample['p0'][0], sample['p0'][1], sample['p0_theta']]
        gt_place = [sample['p1'][0], sample['p1'][1], sample['p1_theta']]

        return image_s1, image_s2, language, gt_pick, gt_place

def save_to_hdf5(hdf5_path, dataset):
    """
    Save a RavensMultiTaskDataset to an HDF5 file.

    Args:
        hdf5_path (str): Path to the HDF5 file.
        dataset (RavensMultiTaskDataset): Dataset instance containing the samples.
    """
    with h5py.File(hdf5_path, 'a') as hdf5_file:
                
        for task in dataset.tasks:

            # Create a group for each task if it doesn't already exist
            if task not in hdf5_file:
                task_group = hdf5_file.create_group(task)
                
                # Create expandable datasets for each type of data
                task_group.create_dataset("image", shape=(0, *dataset.in_shape), maxshape=(None, *dataset.in_shape), dtype='float32', compression="gzip")
                task_group.create_dataset("language", shape=(0,), maxshape=(None,), dtype=h5py.string_dtype(encoding='ascii'))
                task_group.create_dataset("gt_pick", shape=(0, 3), maxshape=(None, 3), dtype='float32')
                task_group.create_dataset("gt_place", shape=(0, 3), maxshape=(None, 3), dtype='float32')
            else:
                task_group = hdf5_file[task]

            images, languages, gt_picks, gt_places = [], [], [], []

            # set task and path
            sample_set = dataset.sample_set[task]
            dataset._task=task
            dataset._path=os.path.join(dataset.root_path, f'{dataset._task}')

            # Load sample and goal
            for episode_id in sample_set:
                # Load sample and goal
                episode, _ = dataset.load(episode_id, True, False)

                # load steps in the episode
                for step in range(len(episode)-1):
                    sample = episode[step]

                    #process step
                    sample = dataset.process_sample(sample, augment=False)
                    image_s1, _, language_encoded, gt_pick, gt_place = interpret(sample, None)

                    # Append each sample to task data lists
                    images.append(image_s1)
                    languages.append(language_encoded)
                    gt_picks.append(gt_pick)
                    gt_places.append(gt_place)
                
            # After accumulating data for the task, resize and write once
            num_samples = len(images)
            current_size = task_group['image'].shape[0]
            new_size = current_size + num_samples

            task_group['image'].resize((new_size, *dataset.in_shape))
            task_group['language'].resize((new_size,))
            task_group['gt_pick'].resize((new_size, 3))
            task_group['gt_place'].resize((new_size, 3))

            # Write all data for this task
            task_group['image'][-num_samples:] = images
            task_group['language'][-num_samples:] = languages
            task_group['gt_pick'][-num_samples:] = gt_picks
            task_group['gt_place'][-num_samples:] = gt_places

            print(f"Task {task} saved to {hdf5_path} successfully.")
            print(f"Task {task} has {new_size} samples.")

    print(f"Data saved to {hdf5_path} successfully.")

@hydra.main(config_path="./cfg", config_name='train')
def main(cfg):
    # Config
    data_dir = cfg['train']['data_dir']
    task = cfg['train']['task']
    n_demos = cfg['train']['n_demos']

    # Datasets
    dataset_type = cfg['dataset']['type']
    if 'multi' in dataset_type:
        ds = RavensMultiTaskDataset(data_dir, cfg, group=task, mode='train', n_demos=n_demos, augment=False)
    else:
        #TODO: save train set, test set and val set separately
        ds = RavensDataset(os.path.join(data_dir, '{}-train'.format(task)), cfg, n_demos=n_demos, augment=False)
  
    # Save to HDF5
    PATH = '/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/data_hdf5/ravens_multi_1000.hdf5'
    save_to_hdf5(PATH, ds)


if __name__ == '__main__':
    main()

