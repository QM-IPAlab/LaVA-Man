"""
load dataset from cliport for mae training
modified from https://github.com/google-research/ravens/blob/master/ravens/dataset.py
"""

from cliport.dataset import RavensDataset

class DatasetMAE(RavensDataset):
    def __getitem__(self, idx):
        pass

