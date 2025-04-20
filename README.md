# JimmyTorch
Some Utility Functions that I often use in deep learning projects
Plus a lightweight pipeline for training models

## Usage

> ### Dataset
> Define your own dataset class that inherits from `JimmyDataset`
1. A `__init__` method that does all data loading and convert all data to tensors, and sets `self.n_samples`.
2. A `__getitem__` method that returns a batch of data stored in a dictionary.

> ### Model
> Define your own neural network class that inherits from `JimmyModel`
1. A `__init__` method just like normal pytorch models, additionally, it must define `self.loss_names`.
2. A `forward` method just like normal pytorch models
3. A `forwardBackward` method that processes model forward, loss computation, loss backward, mixed precision training, and returns loss dictionary and output dictionary.

> ### Training and Experiment
> Please refer to `JimmyTrainer.py` for the training pipeline, and please refer to `JimmyExperiment.py` for the experiment pipeline. Finally, the `main.py` is the entry.

## TODOs
- [ ] Add evaluation into trainer pipeline
- [ ] Add testing into experiment pipeline

