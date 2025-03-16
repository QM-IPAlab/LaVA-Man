"""
Read data for mae training
"""
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import h5py
import torchvision.transforms as transforms
from cliport.utils import utils
import numpy as np
import random
PATH = "/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/data_hdf5/exist_dataset.hdf5"


class MAEDatasetCV(Dataset):
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

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        if self.file is None:
            self.file = h5py.File(self.data_path, 'r')

        img1 = self.file['image_s1'][idx]
        img2 = self.file['image_s2'][idx]
        imgcv = self.file['image_cv'][idx]
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

            img1 = self.transform(img1)
            img2 = self.transform(img2)
            imgcv = self.transform(imgcv)

        if self.condition_free:
            if random.random() < 0.5:
                lang = ''.encode('ascii')
                img2 = img1

        return img1,(img2, imgcv), lang, pick, place


def get_raw_transform():
    trasform_fix = transforms.Compose([
        transforms.ToTensor()])
    return trasform_fix


def main():
    from tqdm import tqdm
    import torch
    dataset = MAEDataset(transform=get_raw_transform())
    dataloader = DataLoader(dataset, batch_size=60, shuffle=False,
                            num_workers=2, drop_last=True)

    mean = 0.0
    sum_sq = 0.0
    nb_samples = 0
    for i, (img1, img2, lang, pick, place) in enumerate(tqdm(dataloader)):
        batch_size = img1.size(0)
        img = img1.view(batch_size, img1.size(1), -1)
        mean += img.mean(2).sum(0)
        sum_sq += ((img - img.mean(2, keepdim=True))**2).sum([0, 2])
        nb_samples += batch_size

    nb_samples = nb_samples * img.size(2)  # 总的像素点数
    mean /= nb_samples
    std = torch.sqrt(sum_sq / nb_samples)

    print("Mean: ", mean)
    print("Std: ", std)


if __name__ == "__main__":
    main()
