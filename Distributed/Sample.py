"""Distributed examples; original SampleCNN and MNISTSampleDataset stay unchanged."""
import torch
import torchvision
from Models.SampleCNN.SampleCNN import SampleCNN
from Models.Functional import getAutoCast
from .Model import DistributedJimmyModel
from .Dataset import DistributedJimmyDataset


class DistributedSampleCNN(SampleCNN, DistributedJimmyModel):
    def trainStep(self, data_dict):
        with getAutoCast(data_dict["input"], self.mixed_precision):
            output = self(data_dict["input"])
            loss = self.ce_loss(output, data_dict["target"])
        self.backwardOptimize(loss)
        return {"Train/Main": loss.item()}, {"output": output.detach()}


class DistributedMNISTDataset(DistributedJimmyDataset):
    @classmethod
    def prepareData(cls, root="Datasets/MNIST", **dataset_args):
        for training in (True, False):
            torchvision.datasets.MNIST(root=root, train=training, download=True)

    def __init__(self, set_name, batch_size, drop_last=False, shuffle=False, root="Datasets/MNIST"):
        super().__init__(set_name, batch_size, drop_last, shuffle)
        if set_name not in ("train", "eval", "test", "debug"):
            raise ValueError(f"Unknown MNIST split: {set_name}")
        dataset = torchvision.datasets.MNIST(root=root, train=set_name != "test", download=False)
        split_index = int(len(dataset) * 0.9)
        selection = {"train": slice(0, split_index), "eval": slice(split_index, None),
                     "test": slice(None), "debug": slice(0, 300)}[set_name]
        # Host storage avoids replicating the full dataset in GPU memory.
        self.images = dataset.data[selection].unsqueeze(1).float().div(255)
        self.labels = dataset.targets[selection]
        self.n_samples = len(self.labels)

    def __getitem__(self, idx):
        indices = self._indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        return {"input": self.images[indices], "target": self.labels[indices], "indices": indices}
