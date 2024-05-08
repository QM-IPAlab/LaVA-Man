"""
Read data for mae training
"""
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import h5py
PATH = "/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/data_hdf5/exist_dataset.hdf5"

class MAEDataset(Dataset):
    def __init__(self, data_path=PATH, transform=None):
        super().__init__()
        self.data_path = data_path
        self.file = None
            # get the length
        with h5py.File(self.data_path, 'r') as f:
            self.length = len(f['image_s1'])
            print("Length of the dataset: ", self.length)
        
        self.transform = transform


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        
        if self.file is None:
            self.file = h5py.File(self.data_path, 'r')

        img1 = self.file['image_s1'][idx]
        img2 = self.file['image_s2'][idx]
        lang = self.file['language'][idx]
        pick = self.file['gt_pick'][idx]
        place = self.file['gt_place'][idx]

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, lang, pick, place


def main():
    dataset = MAEDataset()
    dataloader = DataLoader(dataset, batch_size=20, shuffle=True, 
        num_workers=4, drop_last=True)
    for i, (img1, img2, lang, pick, place) in enumerate(dataloader):
        import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()