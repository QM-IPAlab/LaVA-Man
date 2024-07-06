"""
Save existing dataset to hdf5 format
"""

from cliport.dataset import RavensDataset
import h5py
import os
import numpy as np
from tqdm import tqdm
import hydra


class RavensDatasetToHdf5(RavensDataset):

    def __init__(self, data_path, cfg, n_demos=0, augment=False):
        super().__init__(data_path, cfg, n_demos, augment)

    def save_to_hdf5(self):

        data_s1 = []  #TODO： dynamic save image data ?
        data_s2 = []
        data_language = []
        data_gt_pick = []
        data_gt_place = []

        # TODO: save to different file name
        f = h5py.File(os.path.join('/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/data_hdf5',
                                   'exist_dataset_no_aug_all_test.hdf5'), 'a')

        for idx in range(len(self)):

            # Load the dpisode data determined by the index
            episode, _ = self.load(idx, self.images, self.cache)

            for n_sample in range(len(episode) - 1):
                sample = episode[n_sample]
                goal = episode[n_sample + 1]
                sample = self.process_sample(sample, augment=self.augment)
                goal = self.process_goal(goal, perturb_params=sample['perturb_params'])

                # Interpret the sample and goal data
                image_s1, image_s2, language_encoded, gt_pick, gt_place = self.interpret(sample, goal)
                data_s1.append(image_s1)
                data_s2.append(image_s2)
                data_language.append(language_encoded)
                data_gt_pick.append(gt_pick)
                data_gt_place.append(gt_place)

        #turn to format suitable for hdf5
        data_language = np.array(data_language, dtype=h5py.special_dtype(vlen=str))
        data_s1 = np.array(data_s1)
        data_s2 = np.array(data_s2)
        data_gt_pick = np.array(data_gt_pick)
        data_gt_place = np.array(data_gt_place)

        n1 = self.append_or_create_dataset(f,'image_s1', data=data_s1)
        n2 = self.append_or_create_dataset(f,'image_s2', data=data_s2)
        n3 = self.append_or_create_dataset(f,'language', data=data_language, dtype=h5py.string_dtype(encoding='ascii'))
        n4 = self.append_or_create_dataset(f,'gt_pick', data=data_gt_pick)
        n5 = self.append_or_create_dataset(f,'gt_place', data=data_gt_place, )
        f.close()

        assert n1 == n2 == n3 == n4 == n5

        print(f'Saved {len(data_s1)} samples to the hdf5 file.')
        print(f'Current number of samples in hdf5 file: {n5}.')

    def read_hdf5(self):
        self.data = h5py.File(self.data_file, 'r')

    def append_or_create_dataset(self, f, name, data, dtype=None):
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

    def interpret(self, sample, goal):
        # Interpret the sample and goal data
        image_s1 = sample['img']
        image_s2 = goal['img']
        language = sample['lang_goal']
        gt_pick = [sample['p0'][0], sample['p0'][1], sample['p0_theta']]
        gt_place = [sample['p1'][0], sample['p1'][1], sample['p1_theta']]

        return image_s1[:, :, :3], image_s2[:, :, :3], language, gt_pick, gt_place


@hydra.main(config_path="./cfg", config_name='train')
def main(cfg):
    # Config
    data_dir = cfg['train']['data_dir']
    task = cfg['train']['task']
    n_demos = cfg['train']['n_demos']

    # Datasets
    dataset_type = cfg['dataset']['type']
    if 'multi' in dataset_type:
        ds = RavensDatasetToHdf5(data_dir, cfg, group=task, mode='test', n_demos=n_demos, augment=False)
    else:
        #TODO: save train set, test set and val set separately
        ds = RavensDatasetToHdf5(os.path.join(data_dir, '{}-test'.format(task)), cfg, n_demos=n_demos, augment=False)
    ds.save_to_hdf5()


if __name__ == '__main__':
    main()
