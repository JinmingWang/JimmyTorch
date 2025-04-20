import torch
import torch.nn as nn
from Datasets import *
from Training import *
from Models import JimmyModel, SampleCNN
from datetime import datetime
import os
from rich import print as rprint
from typing import Dict, Any


class JimmyTrainer:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self,
                 dataset: JimmyDataset,
                 model: JimmyModel,
                 optimizer: torch.optim.Optimizer,
                 lr_scheduler: JimmyLRScheduler,
                 comments: str,
                 moving_avg: int = 100,
                 mixed_precision: bool = False,
                 compile_model: bool = False,
                 clip_grad: float = 0.0) -> None:
        """
        Initialize the trainer with a dataset, model, optimizer, and comments.

        :param dataset: A `JimmyDataset` object that provides the training data. It must return a dictionary containing the input data and target labels.
        :param model: A `JimmyModel` object. The model must implement a `forwardBackward` function that returns a dictionary of loss and output, and must have a `loss_names` attribute that lists the keys of the loss dictionary.
        :param optimizer: A PyTorch optimizer used to update the model parameters.
        :param comments: A string containing comments or notes about the training session.
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
        self.lr_scheduler = lr_scheduler
        self.moving_avg = moving_avg
        self.global_step = 0

        # Create save dir and log_dir based on the model name and current time
        now_str = datetime.now().strftime("%y%m%d_%H%M%S")
        self.save_dir = f"Runs/{model.__class__.__name__}/{now_str}/"
        self.log_dir = f"Runs/{model.__class__.__name__}/{now_str}/"

        # Create directories if they do not exist
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Save the model architecture and comments, for reviewing later, or used as a backup
        with open(os.path.join(self.log_dir, "model_arch.txt"), "w") as f:
            f.write(str(model))
        with open(os.path.join(self.log_dir, "comments.txt"), "w") as f:
            f.write(comments)

        rprint(f"[blue]Save directory: {self.save_dir}.[/blue]")
        rprint(f"[blue]Log directory: {self.log_dir}.[/blue]")


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


    def train(self, epochs: int) -> None:
        """
        Train the model for a specified number of epochs.

        :param epochs: The number of epochs to train the model.
        """
        loss_names = self.model.loss_names
        log_tags = loss_names + ["LR"]
        # Initialize progress manager and tensorboard manager
        pm = ProgressManager(self.dataset.n_batches, epochs, 5, 2, custom_fields=log_tags)
        tm = TensorBoardManager(self.log_dir, log_tags, ['scalar'] * len(log_tags))
        ma_losses = {name: MovingAvg(self.moving_avg) for name in loss_names}

        best_loss = float('inf')

        for epoch in range(epochs):
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


if __name__ == '__main__':
    # Example usage
    n_epochs = 100
    dataset: JimmyDataset = MNISTSampleDataset(set_name="train", batch_size=64, drop_last=False, shuffle=True)
    model: JimmyModel = SampleCNN()
    optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    lr_scheduler: JimmyLRScheduler = JimmyLRScheduler(optimizer, init_lr=1e-5, peak_lr=2e-4, min_lr=1e-7,
                                                      warmup_count=10, window_size=10, patience=10, decay_rate=0.5)

    trainer: JimmyTrainer = JimmyTrainer(dataset,
                                         model,
                                         optimizer,
                                         lr_scheduler,
                                         comments="Training with Adam optimizer"
                                         )
    trainer.train(n_epochs)

