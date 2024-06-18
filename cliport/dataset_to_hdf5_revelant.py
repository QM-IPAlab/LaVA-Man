"""
Save existing dataset to hdf5 format
"""
from transformers import CLIPModel
from torchvision import transforms
import torch.nn.functional as F
from transformers import AutoTokenizer
from cliport.dataset import RavensDataset
import h5py
import os
import numpy as np
from tqdm import tqdm
import hydra
from mae.relevance_tools import interpret_ours

MEAN_CLIPORT = [0.48145466, 0.4578275, 0.40821073]
STD_CLIPORT = [0.26862954, 0.26130258, 0.27577711]

class RavensDatasetToHdf5Relevant(RavensDataset):

    def __init__(self, data_path, cfg, n_demos=0, augment=False):
        super().__init__(data_path, cfg, n_demos, augment)

        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.resize_transform = transforms.Resize((224, 224))
        self.text_processor = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.trasform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN_CLIPORT, std=STD_CLIPORT),
            transforms.Pad(padding=(80,0)),
            transforms.Resize((224, 224))
        ])


    def save_to_hdf5(self):

        data = []
        mode = self.cfg.dataset.mode

        # TODO: save to different file name
        f = h5py.File(os.path.join('/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/data_hdf5',
                                   f'exist_dataset_no_aug_all_relevant_map_{mode}.hdf5'), 'a')

        for idx in range(len(self)):

            # Load the dpisode data determined by the index
            episode, _ = self.load(idx, self.images, self.cache)

            for n_sample in range(len(episode) - 1):
                sample = episode[n_sample]
                goal = episode[n_sample + 1]
                sample = self.process_sample(sample, augment=self.augment)
                goal = self.process_goal(goal, perturb_params=sample['perturb_params'])

                # Interpret the sample and goal data
                import pdb; pdb.set_trace()
                image_s1, image_s2, language_encoded, gt_pick, gt_place = self.interpret(sample, goal)
                processed_lang = generate_token(self.text_processor, language_encoded, 'cuda')
                processed_img = self.trasform(image_s1).unsqueeze(0).to('cuda')
                relevance_map = interpret_ours(processed_img, processed_lang['input_ids'], self.clip, 'cuda')

                dim = int(relevance_map.shape[-1]** 0.5)
                relevance_map = relevance_map.reshape(-1,dim,dim)
                relevance_map = relevance_map.unsqueeze(1)
                relevance_map = F.interpolate(relevance_map, size=320, mode='bilinear')
                relevance_map = relevance_map[:, :, :, 80:-80]

                data.append((relevance_map))



        def generate_token(text_processor, lang, device):
            if type(lang) is str:
                decoded_strings = [lang]
            else:
                decoded_strings = [s.decode('ascii') for s in lang]
            processed_lang = text_processor(text=decoded_strings, padding="max_length", return_tensors='pt')
            processed_lang = processed_lang.to(device)
            return processed_lang


        def append_or_create_dataset(name, data, dtype=None):

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

        #turn to format suitable for hdf5
        data_language = np.array(data_language, dtype=h5py.special_dtype(vlen=str))
        data_s1 = np.array(data_s1)
        data_s2 = np.array(data_s2)
        data_gt_pick = np.array(data_gt_pick)
        data_gt_place = np.array(data_gt_place)

        n1 = append_or_create_dataset('image_s1', data=data_s1)
        n2 = append_or_create_dataset('image_s2', data=data_s2)
        n3 = append_or_create_dataset('language', data=data_language, dtype=h5py.string_dtype(encoding='ascii'))
        n4 = append_or_create_dataset('gt_pick', data=data_gt_pick)
        n5 = append_or_create_dataset('gt_place', data=data_gt_place, )
        f.close()

        assert n1 == n2 == n3 == n4 == n5

        print(f'Saved {len(data_s1)} samples to the hdf5 file.')
        print(f'Current number of samples in hdf5 file: {n5}.')

    def read_hdf5(self):
        self.data = h5py.File(self.data_file, 'r')

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
    data_mode = cfg['dataset']['mode']
    ds = RavensDatasetToHdf5Relevant(os.path.join(data_dir, '{}-{}'.format(task, data_mode)), cfg, n_demos=n_demos, augment=False)
    ds.save_to_hdf5()


if __name__ == '__main__':
    main()
