"""
dataset used to mix different datasets of different size
"""


class SizeBatchSampler:
    def __init__(self, datasets, batch_size):
        self.datasets = datasets
        self.batch_size = batch_size
        self.dataset_offsets = self._compute_offsets()
    
    def _compute_offsets(self):
        # Compute cumulative offsets for indexing
        offsets = [0]
        for ds in self.datasets:
            offsets.append(offsets[-1] + len(ds))
        return offsets

    def __iter__(self):
        batches = []
        for i, ds in enumerate(self.datasets):
            # Create batches for each sub-dataset
            indices = range(self.dataset_offsets[i], self.dataset_offsets[i+1])
            ds_batches = [
                indices[j:j + self.batch_size]
                for j in range(0, len(indices), self.batch_size)
            ]
            batches.extend(ds_batches)
        # Shuffle all batches across datasets
        import random
        random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        total_samples = sum(len(ds) for ds in self.datasets)
        return total_samples // self.batch_size