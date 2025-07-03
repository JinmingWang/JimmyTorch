# JimmyTorch
A personal deep learning experiment tool based on PyTorch, containing many functions related to datasets, models, training, visualization, etc. Since it is for personal use, the content is not complete, and I will continue to add what I need.

[![en](https://img.shields.io/badge/lang-en-blue.svg)](README.md)
[![cn](https://img.shields.io/badge/lang-cn-red.svg)](README.cn.md)

## Usage

> ### Dataset
> Define your own dataset class that inherits from `JimmyDataset`
1. A `__init__` method that does all data loading and convert all data to tensors, and sets `self.n_samples`.
2. A `__getitem__` method that returns a batch of data stored in a dictionary.

> ### Model
> Define your own neural network class that inherits from `JimmyModel`
1. A `__init__` method just like normal pytorch models, additionally, it must define `self.train_loss_names` and `self.eval_loss_names`.
2. A `forward` method just like normal pytorch models
3. A `trainStep` method that processes model forward, loss computation, loss backward, mixed precision training, and returns loss dictionary and output dictionary.
4. A `evalStep` method that processes model forward, loss computation, and returns loss dictionary and output dictionary.

> ### Training and Experiment
> Please refer to `JimmyTrainer.py` for the training pipeline, and please refer to `JimmyExperiment.py` for the experiment pipeline. Finally, the `main.py` is the entry.

## Module Introduction

> ### Datasets
- `JimmyDataset.py`: Base class for datasets, you can define your own dataset by inheriting this class. This is more like a combination of Dataset + DataLoader in PyTorch. In contrast to PyTorch's data loading process, this class assumes that the data has already been processed into tensors and loaded to GPU, so the `__getitem__` method can directly use range indexing to access the data, which can be more efficient than multi-threaded data loading. The `__getitem__` method returns a dictionary containing the data, which aims to preserve good consistency with different models and training processes.
- `DatasetUtils.py`: Defines `DEVICE` and also import some useful functions for dataset processing.
- `MultiThreadLoader.py`: In case if the data processing do contain some heavy cpu computation, you can use this class to load data in a multi-threaded way.
- `TrajectoryUtils.py`: My research is mainly about GPS trajectories, so this file contains some utility functions for trajectory processing, such as trajectory interpolation, compute distances, cropping, padding, flipping, etc.
- `SequenceUtils.py`: TODO, some utility functions for sequential data processing.
- `TODO`: In the future, I may add some more utility functions for other types of data, such as images, texts, graphs, etc.

> ### Models
- `JimmyModel.py`: Base class for models, you can define your own model by inheriting this class. This class integrates the training, loss computing, optimization and evaluation process, and also supports mixed precision training and torch.compile. The `train_loss_names`, `eval_loss_names` are used to tell the types of losses and metrics that the model will report during training and evaluation, which is useful for logging and visualization. Then, the `trainStep` and `evalStep` methods are used to process an individual training or evaluation step, which returns a dictionary for all losses, another dictionary for all outputs. The dictionary for all losses must match the `train_loss_names` and `eval_loss_names`. This module aims to keep the uniqueness of different models within the model definition, so that the training and evaluation process can be more reusable and consistent across different models.
- `Basics.py`: Contains some very basic modules, such as combinations of norm + activation + convolution, MLP, positional encoding, make patchs.
- `Attentions.py`: Contains some attention modules, such as MHSA, CrossAttention, and SELayers.
- `ModelUtils.py`: Contains some non-parametric modules, such as Transpose, Permute, Reshape, PrintShape, SequentialMultiIO, Rearrange.

> ### Training and Experiment
- `ProgressManager.py`: A mush better version of progress visualization tool.
- `MovingAverage.py`: Computes moving average of a scalar.
- `TensorBoardManager.py`: A wrapper for TensorBoard, which supports pre-registered tags to log.
- `JimmyTrainer.py`: An example trainer class.
- `JimmyExperiment.py`: An example experiment class.
- `DynamicConfig.py`: Supports a config with a class and its initialization parameters, it can call `build` method to create the class instance.

