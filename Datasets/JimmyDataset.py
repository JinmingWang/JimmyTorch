from .DatasetUtils import *

class JimmyDataset():
    """
    JimmyDataset defines a dataset format that compatible with many different datasets.

    Most importantly, the __getitem__ function returns a dictionary, this is helpful for unifying the training code
    when you want to use different datasets for different models in your experiments.

    """
    def __init__(self, batch_size: int, drop_last: bool = False, shuffle: bool = False):
        """
        :param batch_size: number of samples per batch
        :param drop_last: if True, drop the last batch if it is smaller than batch_size
        :param shuffle: if True, shuffle the dataset at the beginning of each epoch
        """
        rprint("[blue]Initializing dataset[/blue]")
        self.batch_size = batch_size
        self.n_samples = 100
        self.drop_last = drop_last
        self.shuffle = shuffle

    @property
    def n_batches(self) -> int:
        """ Because n_samples may be set later, we need to recalculate n_batches every time """
        if self.drop_last:
            return self.n_samples // self.batch_size
        else:
            return (self.n_samples + self.batch_size - 1) // self.batch_size

    def __len__(self) -> int:
        return self.n_samples


    def __iter__(self):
        self.iter_idx = 0
        if self.shuffle:
            self._indices = torch.randperm(self.n_samples)
        else:
            self._indices = torch.arange(self.n_samples)
        return self


    def __next__(self):
        if self.iter_idx >= self.n_batches:
            raise StopIteration
        self.iter_idx += 1
        return self.__getitem__(self.iter_idx)

    def __getitem__(self, idx):
        start = (idx - 1) * self.batch_size
        end = min(start + self.batch_size, self.n_samples)
        indices = self.__indices[start:end]
        return {
            'indices': indices,
            'input': torch.randn(len(indices), 10),
            'target': torch.randint(0, 2, (len(indices),))
        }
