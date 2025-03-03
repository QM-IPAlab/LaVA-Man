import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data import Sampler, DistributedSampler
import numpy as np
from PIL import Image
import random

import torchvision.transforms as transforms
from torch.nn.utils.rnn import pad_sequence

# 3️ 自定义 BatchSampler，确保 batch 只从一个数据集里取
class SameDatasetBatchSampler(Sampler):
    def __init__(self, datasets, batch_size, drop_last=False):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.datasets = datasets
        self.lens = [len(d) for d in datasets]

        # 计算数据集的索引范围
        self.dataset_ranges = np.cumsum([0] + self.lens)

        # 计算 batch 索引
        self.batch_indices = self._create_batches()

    def _create_batches(self):
        batch_indices = []
        for dataset_idx, (start, end) in enumerate(zip(self.dataset_ranges[:-1], self.dataset_ranges[1:])):
            indices = np.arange(start, end)
            np.random.shuffle(indices)  # 打乱数据

            # 生成 batch
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue  # 丢弃最后一个 batch
                batch_indices.append(batch)

        np.random.shuffle(batch_indices)  # 打乱 batch 顺序，保证不同数据集的 batch 交替出现
        return batch_indices

    def __iter__(self):
        for batch in self.batch_indices:
            yield batch

    def __len__(self):
        return len(self.batch_indices)


class DistributedSameDatasetBatchSampler(Sampler):
    def __init__(self, datasets, batch_size, num_replicas=None, rank=None, drop_last=False):
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_available() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_available() else 0

        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.drop_last = drop_last
        self.datasets = datasets
        self.lens = [len(d) for d in datasets]

        # 计算数据集的索引范围
        self.dataset_ranges = np.cumsum([0] + self.lens)

        # 计算 batch 索引
        self.batch_indices = self._create_batches()
        self.indices = np.concatenate(self.batch_indices).tolist()

        self.epoch = 0

        # # 每个 GPU 取自己的 batch 子集
        # self.total_batches = len(self.batch_indices)
        # self.num_batches_per_rank = self.total_batches // self.num_replicas
        # if not drop_last and self.total_batches % self.num_replicas != 0:
        #     self.num_batches_per_rank += 1  # 处理 uneven case

        # self.batch_indices = self.batch_indices[self.rank:self.total_batches:self.num_replicas]

    def _create_batches(self):
        batch_indices = []
        for dataset_idx, (start, end) in enumerate(zip(self.dataset_ranges[:-1], self.dataset_ranges[1:])):
            indices = np.arange(start, end)
            np.random.shuffle(indices)  # 打乱数据

            # 生成 batch
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue  # 丢弃最后一个 batch
                batch_indices.append(batch)

        np.random.shuffle(batch_indices)  # 打乱 batch 顺序
        return batch_indices

    def __iter__(self):
        indices = self.indices[self.rank::self.num_replicas]
        return iter(indices)

    def __len__(self):
        return len(self.indices)
    
    def set_epoch(self, epoch):
        """重新 shuffle 数据并重新生成 batch"""
        self.batch_indices = self._create_batches()  # 重新生成 batch
        self.epoch = epoch
