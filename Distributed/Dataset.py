import torch
from Datasets.JimmyDataset import JimmyDataset

class DistributedJimmyDataset(JimmyDataset):
    """Batch-index dataset with global length and rank-local iteration.

    Subclasses retain __getitem__(batch_idx), indexing through self._indices.
    All ranks must expose the same logical samples in the same initial order.
    """
    def __init__(self, set_name, batch_size, drop_last=False, shuffle=False):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        super().__init__(set_name, batch_size, drop_last, shuffle)
        self.distributed_rank = 0
        self.distributed_world_size = 1
        self.distributed_seed = 0
        self.epoch = 0

    @classmethod
    def prepareData(cls, **dataset_args):
        """Optional download/preprocessing hook, before constructing any split."""

    def configureDistributed(self, rank=0, world_size=1, seed=0):
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("Invalid distributed rank or world size.")
        self.distributed_rank = rank
        self.distributed_world_size = world_size
        self.distributed_seed = seed

    def setEpoch(self, epoch):
        self.epoch = epoch

    @property
    def local_n_samples(self):
        if self.drop_last:
            return self.n_samples // self.distributed_world_size
        return (self.n_samples + self.distributed_world_size - 1) // self.distributed_world_size

    @property
    def n_batches(self) -> int:
        """ Because n_samples may be set later, we need to recalculate n_batches every time """
        if self.drop_last:
            return self.local_n_samples // self.batch_size
        else:
            return (self.local_n_samples + self.batch_size - 1) // self.batch_size


    def __iter__(self):
        """ Define the iterator so we can use `for batch in dataset` syntax """
        self.iter_idx = 0
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(
            self, num_replicas=self.distributed_world_size, rank=self.distributed_rank,
            shuffle=self.shuffle, seed=self.distributed_seed, drop_last=self.drop_last,
        )
        sampler.set_epoch(self.epoch)
        self._indices = torch.tensor(list(sampler), dtype=torch.long)
        return self

