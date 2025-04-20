from .DatasetUtils import *
from .JimmyDataset import JimmyDataset

class MNISTSampleDataset(JimmyDataset):
    """
    MNIST dataset for testing purposes.
    """
    def __init__(self,
                 set_name: Literal["train", "eval", "test", "debug"],
                 batch_size: int,
                 drop_last: bool = False,
                 shuffle: bool = False):
        super().__init__(batch_size, drop_last, shuffle)
        self.set_name = set_name
        self.dataset = torchvision.datasets.MNIST(root='Datasets/MNIST', train=(set_name in ["train", "eval", "debug"]), download=True)
        self.n_samples = len(self.dataset)

        self.__images = torch.zeros(self.n_samples, 1, 28, 28)
        self.__labels = torch.zeros(self.n_samples, dtype=torch.long)
        for i in range(self.n_samples):
            image, label = self.dataset[i]
            # image is a PIL image, convert it to a tensor
            self.__images[i] = torchvision.transforms.functional.to_tensor(image)
            self.__labels[i] = label
            # self.__images[i], self.__labels[i] = self.dataset[i]

        # Convert images and labels to tensors
        self.__images = self.__images.to(DEVICE)
        self.__labels = self.__labels.to(DEVICE)

        match set_name:
            case "train":
                self.n_samples = int(self.n_samples * 0.9)
                self.__images = self.__images[:self.n_samples]
                self.__labels = self.__labels[:self.n_samples]
            case "eval":
                self.n_samples = int(self.n_samples * 0.1)
                self.__images = self.__images[self.n_samples:]
                self.__labels = self.__labels[self.n_samples:]
            case "test":
                pass    # use the whole dataset (note that the test set is loaded)
            case "debug":
                self.n_samples = 300
                self.__images = self.__images[:self.n_samples]
                self.__labels = self.__labels[:self.n_samples]
            case _:
                raise ValueError(f"Unknown set name: {set_name}")


    def __getitem__(self, idx):
        start = (idx - 1) * self.batch_size
        end = min(start + self.batch_size, self.n_samples)
        indices = self._indices[start:end]
        return {
            'indices': indices,
            'input': self.__images[indices],
            'target': self.__labels[indices]
        }
