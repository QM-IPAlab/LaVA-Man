"""
Read data for mae training
"""
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import h5py
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from cliport.utils import utils
import numpy as np
import random
PATH = "/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/data_hdf5/exist_dataset.hdf5"


class MAEDatasetCVDf(Dataset):
    def __init__(self, data_path=PATH, transform=None, aug=False, condition_free=False):
        super().__init__()
        self.data_path = data_path
        self.file = None
        # get the length
        with h5py.File(self.data_path, 'r') as f:
            self.length = len(f['image_s1'])
            print("Length of the dataset: ", self.length)

        self.transform = transform
        self.aug = aug
        self.condition_free = condition_free
        if self.condition_free:
            print("Condition free training")

        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224,224), interpolation=InterpolationMode.NEAREST)])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        if self.file is None:
            self.file = h5py.File(self.data_path, 'r')

        img1 = self.file['image_s1'][idx]
        img2 = self.file['image_s2'][idx]
        imgcv = self.file['image_cv'][idx]
        mask_s1 = self.file['mask_s1'][idx]
        mask_cv = self.file['mask_cv'][idx]
        lang = self.file['language'][idx]
        pick = self.file['gt_pick'][idx] if 'gt_pick' in self.file else 0.0
        place = self.file['gt_place'][idx] if 'gt_pick' in self.file else 0.0

        if self.aug:
            angle = np.random.choice([0, 15, 30])
            img1, _, (p0, p1), pert_urb_params = utils.perturb(img1, (pick, place), theta_sigma=angle)
            pick = p0
            place = p1

        if self.transform:
            if img1.max() > 1:
                img1 = img1 / 255.0
                img2 = img2 / 255.0
                imgcv = imgcv / 255.0
                mask_s1 = mask_s1 / 255.0
                mask_cv = mask_cv / 255.0

            img1 = self.transform(img1)
            img2 = self.transform(img2)
            imgcv = self.transform(imgcv)
            mask_s1 = self.mask_transform(mask_s1)
            mask_cv = self.mask_transform(mask_cv)

        if self.condition_free:
            if random.random() < 0.5:
                lang = ''.encode('ascii')
                img2 = img1

        return (img1, mask_s1), (img2, imgcv, mask_cv), lang, pick, place
