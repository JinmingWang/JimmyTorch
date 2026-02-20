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
import yaml


class JimmyTrainer:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self,
                 train_set: JimmyDataset,
                 eval_set: JimmyDataset,
                 model: JimmyModel,
                 lr_scheduler: JimmyLRScheduler | torch.optim.lr_scheduler._LRScheduler,
                 log_dir: str,
                 save_dir: str,
                 n_epochs: int,
                 moving_avg: int,
                 eval_interval: int = 1,
                 early_stop_lr: float = 0.0) -> None:
        """
        Initialize the trainer with a dataset, model, optimizer, and comments.

        :param train_set: A `JimmyDataset` object that provides the training data. It must return a dictionary containing the input data and target labels.
        :param model: A `JimmyModel` object. The model must implement a `forwardBackward` function that returns a dictionary of loss and output, and must have a `loss_names` attribute that lists the keys of the loss dictionary.
        :param optimizer: A PyTorch optimizer used to update the model parameters.
        :param lr_scheduler: A learning rate scheduler. It can be a `JimmyLRScheduler` or a PyTorch learning rate scheduler. The scheduler must implement an `update` method that takes the current loss as an argument.
        :param log_dir: A string specifying the directory where the training logs will be saved.
        :param save_dir: A string specifying the directory where the model checkpoints will be saved.
        :param moving_avg: An integer specifying the window size for calculating the moving average of the loss. Default is 100.
        :param eval_interval: An integer specifying the interval (in epochs) at which to evaluate the model on the validation set.
        :param early_stop_lr: A float specifying the learning rate threshold for early stopping. Default is 0.0 (no early stopping).
        """

        self.train_set = train_set
        self.eval_set = eval_set
        self.model = model

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
        self.eval_interval = eval_interval
        self.early_stop_lr = early_stop_lr


    def start(self) -> None:
        """
        Train the model for a specified number of epochs.

        :param epochs: The number of epochs to train the model.
        """
        pm_log_tags = self.model.train_loss_names + ["LR"]
        tm_log_tags = self.model.train_loss_names + self.model.eval_loss_names + ["LR"]
        # Initialize progress manager and tensorboard manager
        pm = ProgressManager(self.train_set.n_batches, self.n_epochs, 5, 2, custom_fields=pm_log_tags)
        tm = TensorBoardManager(self.log_dir, tags=tm_log_tags, value_types=["scalar"] * len(tm_log_tags))
        tm.writer.add_text("Comments", "Training Started")
        tm.register("Visualization", "figure")
        ma_losses = {name: MovingAvg(self.moving_avg) for name in self.model.train_loss_names}

        # Runtime parameter buffer for hot-reloading hyperparameters during training
        runtime_param_file = os.path.join(self.log_dir, "runtime_param_buffer.yaml")
        with open(runtime_param_file, "w") as f:
            yaml.dump({"LR": self.model.optimizer.param_groups[0]["lr"]}, f)

        best_loss = float('inf')

        for epoch in range(self.n_epochs):
            loader = MultiThreadLoader(self.train_set, 3)
            for i, data_dict in enumerate(loader):
                # forward, backward, optimization
                loss_dict, output_dict = self.model.trainStep(data_dict)

                # Compute moving average of losses
                for loss_name in self.model.train_loss_names:
                    ma_losses[loss_name].update(loss_dict[loss_name])
                    loss_dict[loss_name] = ma_losses[loss_name].get()

                # Update progress manager
                pm.update(epoch, i, LR=self.model.lr, **loss_dict)

            # Update tensorboard
            tm.log(pm.overall_progress, LR=self.model.lr, **loss_dict)

            # Update learning rate scheduler
            prev_lr = self.model.lr
            self.lr_scheduler.update(loss_dict["Train/Main"])
            
            # Update runtime param file when LR changes
            if self.model.lr != prev_lr:
                with open(runtime_param_file, "w") as f:
                    yaml.dump({"LR": self.model.lr}, f)
            
            # Load runtime parameters (allows user to manually adjust LR during training)
            with open(runtime_param_file, "r") as f:
                runtime_params = yaml.safe_load(f)
                self.model.optimizer.param_groups[0]["lr"] = float(runtime_params["LR"])

            if epoch % self.eval_interval == 0:
                eval_losses = self.evaluate(self.eval_set, pm=pm, tm=tm)

                # Update tensorboard
                tm.log(pm.overall_progress, **eval_losses)

                # Determine the best model based on eval_losses["Eval/Main"]
                eval_loss = eval_losses["Eval/Main"]
                if eval_loss < best_loss:
                    best_loss = eval_loss
                    self.model.saveTo(os.path.join(self.save_dir, "best.pth"))
                self.model.saveTo(os.path.join(self.save_dir, f"last.pth"))

            # Early stopping based on learning rate threshold
            if self.early_stop_lr > 0 and self.model.lr < self.early_stop_lr:
                rprint(f"[red]Learning rate {self.model.lr} is lower than early stop threshold {self.early_stop_lr}. Stopping training.[/red]")
                break

        pm.close()


    def evaluate(self,
                 dataset: JimmyDataset,
                 compute_avg: bool=True,
                 pm: ProgressManager = None,
                 tm: TensorBoardManager = None):
        """
        Evaluate the model on a given dataset.
        
        :param dataset: The dataset to evaluate on.
        :param compute_avg: Whether to compute and return the average loss over the dataset. If False, returns the loss for each batch.
        :param pm: An optional ProgressManager to update during evaluation.
        :param tm: An optional TensorBoardManager to log visualizations during evaluation.
        :return: A dictionary of average losses if compute_avg is True, otherwise a dictionary of loss arrays for each batch.
        """
        n_batches = dataset.n_batches
        # For each type of loss, store a tensor of shape (n_batches,)
        eval_losses = {name: torch.zeros(n_batches).to(DEVICE) for name in self.model.eval_loss_names}
        self.model.eval()

        # Iterate through the dataset and compute losses
        for i, data_dict in enumerate(dataset):
            loss_dict, output_dict = self.model.evalStep(data_dict)
            # Store each loss in the corresponding tensor
            for name in self.model.eval_loss_names:
                eval_losses[name][i] = loss_dict[name]

        # Log visualization if available
        if tm is not None and "fig" in output_dict:
            tm.log(pm.overall_progress, Visualization=output_dict["fig"])

        self.model.train()

        if compute_avg:     # if compute_avg, then average each loss over all batches
            return {name: torch.mean(eval_losses[name]).item() for name in self.model.eval_loss_names}

        return {name: eval_losses[name].cpu().numpy() for name in self.model.eval_loss_names}