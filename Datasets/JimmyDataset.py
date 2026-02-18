from .DatasetUtils import *

class JimmyDataset():
    """
    JimmyDataset defines a dataset format that compatible with many different datasets. All dataset classes should inherit from this class.
    """
    def __init__(self, 
                 set_name: Literal['train', 'eval', 'test', 'debug', 'all'],
                 batch_size: int, 
                 drop_last: bool = False, 
                 shuffle: bool = False):
        """
        The __init__ function of the child class should do the following things:
        1. Call super().__init__() to initialize the dataset.
        2. Do any necessary computing of hyperparameters
        3. Load the processed data (.pth) files into memory, convert to cuda, and apply preprocessing if necessary
        4. Split the data according to set_name
        5. Set self.n_samples to the number of samples

        :param set_name: the name of the dataset split, can be 'train', 'eval', 'test', 'debug', or 'all'
        :param batch_size: number of samples per batch
        :param drop_last: if True, drop the last batch if it is smaller than batch_size
        :param shuffle: if True, shuffle the dataset at the beginning of each epoch
        """
        rprint("[blue]Initializing dataset[/blue]")
        self.set_name = set_name
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
        """ Return the number of samples in the dataset """
        return self.n_samples


    def __iter__(self):
        """ Define the iterator so we can use `for batch in dataset` syntax """
        self.iter_idx = 0
        if self.shuffle:
            self._indices = torch.randperm(self.n_samples)
        else:
            self._indices = torch.arange(self.n_samples)
        return self


    def __next__(self):
        """ Return the next batch of data """
        if self.iter_idx >= self.n_batches:
            raise StopIteration
        batch = self.__getitem__(self.iter_idx)
        self.iter_idx += 1
        return batch

    def __getitem__(self, idx):
        """
        __getitem__ should take a batch index and return a batch of data using dict. The keys are dataset-specific. In order to be compatible with different models that require different input formats, we recommend packing all types of data into this dict (e.g., time-series, spatio-temporal, graph, images, labels, annotations, or even metadata and mean & std used in normalization). The model can then choose which keys to use as input and which keys to use as target.
        """
        start = idx * self.batch_size
        end = min(start + self.batch_size, self.n_samples)
        indices = self._indices[start:end]
        return {
            'indices': indices,
            'input': torch.randn(len(indices), 10),
            'target': torch.randint(0, 2, (len(indices),))
        }
