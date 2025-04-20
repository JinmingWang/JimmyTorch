import torch
import torch.nn as nn
from Datasets import *
from Training import *
from Models import JimmyModel, SampleCNN
from datetime import datetime
import os
from rich import print as rprint
from typing import Dict, Any
import inspect


class JimmyTrainer:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self,
                 dataset: JimmyDataset,
                 model: JimmyModel,
                 optimizer: torch.optim.Optimizer,
                 lr_scheduler: JimmyLRScheduler | torch.optim.lr_scheduler._LRScheduler,
                 log_dir: str,
                 save_dir: str,
                 n_epochs: int = 100,
                 moving_avg: int = 100,
                 mixed_precision: bool = False,
                 compile_model: bool = False,
                 clip_grad: float = 0.0) -> None:
        """
        Initialize the trainer with a dataset, model, optimizer, and comments.

        :param dataset: A `JimmyDataset` object that provides the training data. It must return a dictionary containing the input data and target labels.
        :param model: A `JimmyModel` object. The model must implement a `forwardBackward` function that returns a dictionary of loss and output, and must have a `loss_names` attribute that lists the keys of the loss dictionary.
        :param optimizer: A PyTorch optimizer used to update the model parameters.
        :param lr_scheduler: A learning rate scheduler. It can be a `JimmyLRScheduler` or a PyTorch learning rate scheduler. The scheduler must implement an `update` method that takes the current loss as an argument.
        :param log_dir: A string specifying the directory where the training logs will be saved.
        :param save_dir: A string specifying the directory where the model checkpoints will be saved.
        :param moving_avg: An integer specifying the window size for calculating the moving average of the loss. Default is 100.
        :param mixed_precision: A boolean indicating whether to use mixed precision training. Default is False.
        :param compile_model: A boolean indicating whether to use `torch.compile` to optimize the model. Default is False.
        :param clip_grad: A float specifying the maximum gradient norm for gradient clipping. Default is 0.0 (no clipping).
        """

        self.dataset = dataset
        self.model = model.to(self.device)
        if compile_model:
            self.model: JimmyModel = torch.compile(self.model)
        self.compile_model = compile_model
        self.mixed_precision = mixed_precision
        self.scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
        self.clip_grad = clip_grad
        self.optimizer = optimizer

        if not hasattr(lr_scheduler, 'update'):
            # get number of arguments of lr_scheduler.step()
            num_args = len(inspect.signature(lr_scheduler.step).parameters)
            if num_args == 1:
                lr_scheduler.update = lambda metric: lr_scheduler.step()
            else:
                lr_scheduler.update = lambda metric: lr_scheduler.step(metric)

        self.lr_scheduler = lr_scheduler
        self.log_dir = log_dir
        self.save_dir = save_dir
        self.n_epochs = n_epochs
        self.moving_avg = moving_avg
        self.global_step = 0


    def trainStep(self, data_dict: Dict[str, Any]) -> (Dict[str, float], Dict[str, Any]):
        """
        Perform a single training step.

        :param data_dict: A dictionary containing input data and target labels.
        :return: A tuple containing:
                 - loss_dict: A dictionary of loss values.
                 - output_dict: A dictionary of model outputs.
        """
        loss_dict, output_dict = self.model.forwardBackward(data_dict, self.scaler)
        if self.mixed_precision:
            if self.clip_grad > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            if self.clip_grad > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
            self.optimizer.step()

        self.optimizer.zero_grad()
        self.global_step += 1

        return loss_dict, output_dict


    def start(self) -> None:
        """
        Train the model for a specified number of epochs.

        :param epochs: The number of epochs to train the model.
        """
        loss_names = self.model.loss_names
        log_tags = loss_names + ["LR"]
        # Initialize progress manager and tensorboard manager
        pm = ProgressManager(self.dataset.n_batches, self.n_epochs, 5, 2, custom_fields=log_tags)
        tm = TensorBoardManager(self.log_dir, log_tags, ['scalar'] * len(log_tags))
        ma_losses = {name: MovingAvg(self.moving_avg) for name in loss_names}

        best_loss = float('inf')

        for epoch in range(self.n_epochs):
            loader = MultiThreadLoader(self.dataset, 3)
            for i, data_dict in enumerate(loader):
                loss_dict, output_dict = self.trainStep(data_dict)
                for loss_name in loss_names:
                    ma_losses[loss_name].update(loss_dict[loss_name])
                    loss_dict[loss_name] = ma_losses[loss_name].get()

                pm.update(epoch, i, LR=self.optimizer.param_groups[0]['lr'], **loss_dict)

            tm.log(self.global_step, LR=self.optimizer.param_groups[0]['lr'], **loss_dict)
            loss_sum = sum([ma_losses[name].get() for name in loss_names])
            self.lr_scheduler.update(loss_sum)
            if loss_sum < best_loss:
                best_loss = loss_sum
                saveModels(os.path.join(self.save_dir, "best.pth"), model=self.model)
            saveModels(os.path.join(self.save_dir, "last.pth"), model=self.model)

