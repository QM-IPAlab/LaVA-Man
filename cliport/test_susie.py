"""
Save existing dataset to hdf5 format
"""
from torch.utils.data import Dataset
from cliport.dataset import RavensMultiTaskDataset
import os
import numpy as np
from tqdm import tqdm
import hydra
import pickle
import cv2
from matplotlib import pyplot as plt
import warnings
from susie.model import create_sample_fn
from susie.jax_utils import initialize_compilation_cache
from cliport.tasks import cameras
from cliport.utils import utils

import asyncio

# Save original
_orig_cond_init = asyncio.Condition.__init__

def _patched_condition_init(self, lock=None, *args, **kwargs):
    if lock is not None:
        # Workaround for bpo-45416: ensure the lock's _loop matches
        getattr(lock, '_get_loop', lambda: None)()
    # Now call the real initializer
    return _orig_cond_init(self, lock=lock, *args, **kwargs)

# Apply the patch
asyncio.Condition.__init__ = _patched_condition_init


class RavensDatasetSuSIE(Dataset):
    """A simple image dataset class."""

    def __init__(self, path, cfg, n_demos=0, augment=False):
        """A simple RGB-D image dataset."""
        self._path = path

        self.cfg = cfg
        self.sample_set = []
        self.max_seed = -1
        self.n_episodes = 0
        self.images = self.cfg['dataset']['images']
        self.cache = self.cfg['dataset']['cache']
        self.n_demos = n_demos
        self.augment = augment
        print(f"Augment: {self.augment}")

        self.aug_theta_sigma = self.cfg['dataset']['augment']['theta_sigma'] if 'augment' in self.cfg['dataset'] else 60  # legacy code issue: theta_sigma was newly added
        self.pix_size = 0.003125
        self.in_shape = (320, 160, 6)
        self.cam_config = cameras.RealSenseD415.CONFIG
        self.bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])

        np.random.seed(42)

        # Track existing dataset if it exists.
        color_path = os.path.join(self._path, 'action')
        if os.path.exists(color_path):
            for fname in sorted(os.listdir(color_path)):
                if '.pkl' in fname:
                    seed = int(fname[(fname.find('-') + 1):-4])
                    self.n_episodes += 1
                    self.max_seed = max(self.max_seed, seed)

        self._cache = {}
        if self.n_demos > 0:
            self.images = self.cfg['dataset']['images']
            self.cache = self.cfg['dataset']['cache']

            # Check if there sufficient demos in the dataset
            if self.n_demos > self.n_episodes:
                raise Exception(f"Requested training on {self.n_demos} demos, but only {self.n_episodes} demos exist in the dataset path: {self._path}.")

            # idx from 0 to n_demos-1 in order
            episodes = np.arange(self.n_demos)

            self.set(episodes)

    def add(self, seed, episode):
        """Add an episode to the dataset.

        Args:
          seed: random seed used to initialize the episode.
          episode: list of (obs, act, reward, info) tuples.
        """
        color, depth, action, reward, info = [], [], [], [], []
        for obs, act, r, i in episode:
            color.append(obs['color'])
            depth.append(obs['depth'])
            action.append(act)
            reward.append(r)
            info.append(i)

        color = np.uint8(color)
        depth = np.float32(depth)

        def dump(data, field):
            field_path = os.path.join(self._path, field)
            if not os.path.exists(field_path):
                os.makedirs(field_path)
            fname = f'{self.n_episodes:06d}-{seed}.pkl'  # -{len(episode):06d}
            with open(os.path.join(field_path, fname), 'wb') as f:
                pickle.dump(data, f)

        dump(color, 'color')
        dump(depth, 'depth')
        dump(action, 'action')
        dump(reward, 'reward')
        dump(info, 'info')

        self.n_episodes += 1
        self.max_seed = max(self.max_seed, seed)

    def set(self, episodes):
        """Limit random samples to specific fixed set."""
        self.sample_set = episodes

    def load(self, episode_id, images=True, cache=False):
        """Load data from a saved episode.
        
        Args:
            episode_id: the ID of the episode to be loaded.
            images: load image data if True.
            cache: load data from memory if True.

        Returns:
            episode: list of (obs, act, reward, info) tuples.
            seed: random seed used to initialize the episode.
        """
        
        
        def load_field(episode_id, field, fname):

            # Check if sample is in cache.
            if cache:
                if episode_id in self._cache:
                    if field in self._cache[episode_id]:
                        return self._cache[episode_id][field]
                else:
                    self._cache[episode_id] = {}

            # Load sample from files.
            path = os.path.join(self._path, field)
            data = pickle.load(open(os.path.join(path, fname), 'rb'))
            if cache:
                self._cache[episode_id][field] = data
            return data

        # Get filename and random seed used to initialize episode.
        seed = None
        path = os.path.join(self._path, 'action')
        for fname in sorted(os.listdir(path)):
            if f'{episode_id:06d}' in fname:
                seed = int(fname[(fname.find('-') + 1):-4])

                # Load data.
                color = load_field(episode_id, 'color', fname)
                depth = load_field(episode_id, 'depth', fname)
                action = load_field(episode_id, 'action', fname)
                reward = load_field(episode_id, 'reward', fname)
                info = load_field(episode_id, 'info', fname)
                prediction = load_field(episode_id, 'prediction', fname)

                # Reconstruct episode.
                episode = []
                for i in range(len(action)):
                    obs = {'color': color[i], 'depth': depth[i]} if images else {}
                    episode.append((obs, action[i], reward[i], info[i]))
                return episode, seed, prediction
            
    def load_wo_prediction(self, episode_id, images=True, cache=False):
        """Load data from a saved episode.
        
        Args:
            episode_id: the ID of the episode to be loaded.
            images: load image data if True.
            cache: load data from memory if True.

        Returns:
            episode: list of (obs, act, reward, info) tuples.
            seed: random seed used to initialize the episode.
        """
        
        
        def load_field(episode_id, field, fname):

            # Check if sample is in cache.
            if cache:
                if episode_id in self._cache:
                    if field in self._cache[episode_id]:
                        return self._cache[episode_id][field]
                else:
                    self._cache[episode_id] = {}

            # Load sample from files.
            path = os.path.join(self._path, field)
            data = pickle.load(open(os.path.join(path, fname), 'rb'))
            if cache:
                self._cache[episode_id][field] = data
            return data

        # Get filename and random seed used to initialize episode.
        seed = None
        path = os.path.join(self._path, 'action')
        for fname in sorted(os.listdir(path)):
            if f'{episode_id:06d}' in fname:
                seed = int(fname[(fname.find('-') + 1):-4])

                # Load data.
                color = load_field(episode_id, 'color', fname)
                depth = load_field(episode_id, 'depth', fname)
                action = load_field(episode_id, 'action', fname)
                reward = load_field(episode_id, 'reward', fname)
                info = load_field(episode_id, 'info', fname)

                # Reconstruct episode.
                episode = []
                for i in range(len(action)):
                    obs = {'color': color[i], 'depth': depth[i]} if images else {}
                    episode.append((obs, action[i], reward[i], info[i]))
                return episode, seed

    def get_image(self, obs, prediction, cam_config=None):
        """Stack color and height images image."""

        # if self.use_goal_image:
        #   colormap_g, heightmap_g = utils.get_fused_heightmap(goal, configs)
        #   goal_image = self.concatenate_c_h(colormap_g, heightmap_g)
        #   input_image = np.concatenate((input_image, goal_image), axis=2)
        #   assert input_image.shape[2] == 12, input_image.shape

        if cam_config is None:
            cam_config = self.cam_config

        # Get color and height maps from RGB-D images.
        cmap, hmap = utils.get_fused_heightmap(
            obs, cam_config, self.bounds, self.pix_size)
        if prediction is None:
            img = np.concatenate((cmap,
                              hmap[Ellipsis, None],
                              hmap[Ellipsis, None],
                              hmap[Ellipsis, None]), axis=2)
        else:
            img = np.concatenate((cmap, prediction), axis=2)
        assert img.shape == self.in_shape, img.shape
        return img

    def process_sample(self, datum, prediction, augment=False):
        # Get training labels from data sample.
        (obs, act, _, info) = datum
        img = self.get_image(obs, prediction)

        p0, p1 = None, None
        p0_theta, p1_theta = None, None
        perturb_params =  None

        if act:
            p0_xyz, p0_xyzw = act['pose0']
            p1_xyz, p1_xyzw = act['pose1']
            p0 = utils.xyz_to_pix(p0_xyz, self.bounds, self.pix_size)
            p0_theta = -np.float32(utils.quatXYZW_to_eulerXYZ(p0_xyzw)[2])
            p1 = utils.xyz_to_pix(p1_xyz, self.bounds, self.pix_size)
            p1_theta = -np.float32(utils.quatXYZW_to_eulerXYZ(p1_xyzw)[2])
            p1_theta = p1_theta - p0_theta
            p0_theta = 0

        # Data augmentation.
        if augment:
            img, _, (p0, p1), perturb_params = utils.perturb(img, [p0, p1], theta_sigma=self.aug_theta_sigma)

        sample = {
            'img': img,
            'p0': p0, 'p0_theta': p0_theta,
            'p1': p1, 'p1_theta': p1_theta,
            'perturb_params': False,
            'pick_radius': 5.0,
            'place_radius': 5.0
        }

        # Add language goal if available.
        if 'lang_goal' not in info:
            warnings.warn("No language goal. Defaulting to 'task completed.'")

        if info and 'lang_goal' in info:
            sample['lang_goal'] = info['lang_goal']
        else:
            sample['lang_goal'] = "task completed."

        return sample

    def process_goal(self, goal, perturb_params):
        # Get goal sample.
        (obs, act, _, info) = goal
        img = self.get_image(obs)

        p0, p1 = None, None
        p0_theta, p1_theta = None, None

        # Data augmentation with specific params.
        if perturb_params:
            img = utils.apply_perturbation(img, perturb_params)

        sample = {
            'img': img,
            'p0': p0, 'p0_theta': p0_theta,
            'p1': p1, 'p1_theta': p1_theta,
            'perturb_params': perturb_params
        }

        # Add language goal if available.
        if 'lang_goal' not in info:
            warnings.warn("No language goal. Defaulting to 'task completed.'")

        if info and 'lang_goal' in info:
            sample['lang_goal'] = info['lang_goal']
        else:
            sample['lang_goal'] = "task completed."

        return sample

    def __len__(self):
        return len(self.sample_set)

    def __getitem__(self, idx):
        # Choose random episode.
        if len(self.sample_set) > 0:
            episode_id = np.random.choice(self.sample_set)
        else:
            episode_id = np.random.choice(range(self.n_episodes))
        episode, _, prediction = self.load(episode_id, self.images, self.cache)

        # Is the task sequential like stack-block-pyramid-seq?
        #is_sequential_task = '-seq' in self._path.split("/")[-1]

        # Return random observation action pair (and goal) from episode.
        i = np.random.choice(range(len(episode)-1))
        #g = i+1 if is_sequential_task else -1
        sample = episode[i]

        # Process sample.
        sample = self.process_sample(sample, prediction, augment=self.augment)
        #goal = self.process_goal(goal, perturb_params=sample['perturb_params'])

        return sample, sample

    def save_prediction_to_pickle(self, sample_fn):
        
        for idx in tqdm(range(len(self))):
           
            # Load the dpisode data determined by the index
            episode, seed = self.load_wo_prediction(idx, self.images, self.cache)

            #for n_sample in range(len(episode) - 1):
            n_sample = 0
            sample = episode[n_sample]
            
            # Process the sample
            sample = self.process_sample(sample, prediction=None, augment=self.augment)
            image = sample['img']
            language = sample['lang_goal']
            image = image[:, :, :3]
            image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
            image = image.astype(np.uint8)
            # Generate prediction using the sample function
            prediction = sample_fn(image, language, prompt_w=7.5, context_w=1.5)
            prediction = cv2.resize(prediction, (160, 320), interpolation=cv2.INTER_AREA)
            prediction = np.uint8(prediction)

            import pdb; pdb.set_trace()
            from matplotlib import pyplot as plt
            plt.imshow(image)
            plt.show()

            plt.imshow(prediction)
            plt.show()

            # Save the prediction to a pickle file
            field_path = os.path.join(self._path, 'predictions')
            if not os.path.exists(field_path):
                os.makedirs(field_path)
            fname = f'{idx:06d}-{seed}.pkl'
            with open(os.path.join(field_path, fname), 'wb') as f:
                pickle.dump(prediction, f)

class RavensMultiTaskDatasetSuSIE(RavensMultiTaskDataset):
    def __init__(self, data_dir, cfg, group, mode, n_demos, augment):
        super().__init__(data_dir, cfg, group, mode, n_demos, augment)
        print("Data dir: ", data_dir)
        print("Data type: Multi")

    def run(self):

        data_s1 = []  #TODO： dynamic save image data ?
        data_s2 = []
        data_language = []
        data_gt_pick = []
        data_gt_place = []

        # TODO: save to different file name
        print(f'Processing {len(self)} episodes... ')
        for idx in tqdm(range(len(self))):
           
            # Load the dpisode data determined by the index
            episode, _ = self.load(idx, self.images, self.cache)

            for n_sample in range(len(episode) - 1):
                sample = episode[n_sample]
                goal = episode[n_sample + 1]
                assert self.augment == False 
                sample = self.process_sample(sample, augment=self.augment)
                goal = self.process_goal(goal, perturb_params=None)

                # Interpret the sample and goal data
                image_s1, image_s2, language_encoded, gt_pick, gt_place = self.interpret(sample, goal)
                data_s1.append(image_s1)
                data_s2.append(image_s2)
                data_language.append(language_encoded)
                data_gt_pick.append(gt_pick)
                data_gt_place.append(gt_place)

                break

            break

        return data_s1, data_s2, data_language, data_gt_pick, data_gt_place

    def interpret(self, sample, goal):
        # Interpret the sample and goal data
        image_s1 = sample['img']
        image_s2 = goal['img']
        language = sample['lang_goal']
        gt_pick = [sample['p0'][0], sample['p0'][1], sample['p0_theta']]
        gt_place = [sample['p1'][0], sample['p1'][1], sample['p1_theta']]
        return image_s1, image_s2, language, gt_pick, gt_place

@hydra.main(config_path="./cfg", config_name='train')
def main(cfg):
    # Config
    data_dir = "/home/robot/Repositories_chaoran/CLIPort_new_loss/data"
    task = "packing-seen-google-objects-group"
    n_demos = 100

    # Datasets
    dataset_type = "single"
    if 'multi' in dataset_type:
        ds = RavensMultiTaskDatasetSuSIE(data_dir, cfg, group=task, mode='train', n_demos=n_demos, augment=False)
    else:
        #TODO: save train set, test set and val set separately
        ds = RavensDatasetSuSIE(os.path.join(data_dir, '{}-train'.format(task)), cfg, n_demos=n_demos, augment=False)
    
    
    initialize_compilation_cache()
    #sample_fn = create_sample_fn("kvablack/susie")
    sample_fn = create_sample_fn(
        "/home/robot/Repositories_chaoran/CLIPort_new_loss/checkpoints/susie_diffusion",
        "kvablack/dlimp-diffusion/9n9ped8m",
        num_timesteps=50,
        prompt_w=7.5,
        context_w=1.5,
        eta=0.0
        )
    
    ds.save_prediction_to_pickle(sample_fn)
        
    """
    data_s1, data_s2, data_language, data_gt_pick, data_gt_place = ds.run()
    image = data_s1[0][:160, :160, :3]
    
    image = image.astype(np.uint8)
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    plt.imshow(image)
    plt.show()
    

    initialize_compilation_cache()
    #sample_fn = create_sample_fn("kvablack/susie")
    sample_fn = create_sample_fn(
        "/home/robot/Repositories_chaoran/CLIPort_new_loss/checkpoints/susie_diffusion",
        "kvablack/dlimp-diffusion/9n9ped8m",
        num_timesteps=50,
        prompt_w=7.5,
        context_w=1.5,
        eta=0.0
        )
    image_out = sample_fn(image, "open the drawer", prompt_w=7.5, context_w=1.5)
    plt.imshow(image_out)
    plt.show()
    image_out = sample_fn(image, "open the drawer", prompt_w=7.5, context_w=1.5)
    """

if __name__ == '__main__':
    main()
