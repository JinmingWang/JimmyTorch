import torch
import torch.nn as nn
from Datasets import *
from Training import *
from Training.ExperimentManager import (
    ExperimentLogger,
    ExperimentManagerClient,
    STATUS_DONE,
    STATUS_ERROR,
    STATUS_EVALUATING,
    STATUS_TRAINING,
)
from Models import JimmyModel, SampleCNN
import matplotlib.pyplot as plt
from datetime import datetime
import os
from rich import print as rprint
from typing import Dict, Any, Optional
import inspect


class JimmyTrainer:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self,
                 train_set: JimmyDataset,
                 eval_set: JimmyDataset,
                 model: JimmyModel,
                 lr_scheduler: Any,
                 log_dir: str,
                 save_dir: str,
                 n_epochs: int,
                 moving_avg: int,
                 eval_interval: int = 1,
                 early_stop_lr: float = 0.0,
                 progress_show_recent_steps: int = 200,
                 progress_refresh_interval: float = 1.0,
                 progress_host: str = "127.0.0.1",
                 progress_port: int = 9000,
                 exp_logger: Optional[ExperimentLogger] = None,
                 enable_tensorboard: bool = False) -> None:
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

        :param progress_show_recent_steps: show recent steps in GUI
        :param progress_refresh_interval: refresh interval for GUI (in seconds)
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
        self.progress_show_recent_steps = progress_show_recent_steps
        self.progress_refresh_interval = progress_refresh_interval
        self.progress_host = progress_host
        self.progress_port = progress_port
        self.exp_logger = exp_logger
        self.enable_tensorboard = enable_tensorboard


    def start(self) -> None:
        """
        Train the model for a specified number of epochs.

        :param epochs: The number of epochs to train the model.
        """
        pm_log_tags = self.model.train_loss_names + ["LR"]
        tm_log_tags = self.model.train_loss_names + self.model.eval_loss_names + ["LR"]

        run_dir = self.exp_logger.run_dir if self.exp_logger is not None else self.save_dir
        run_meta = {
            "dataset_name": (self.exp_logger.dataset_name if self.exp_logger is not None else ""),
            "model_name": (self.exp_logger.model_name if self.exp_logger is not None else ""),
            "run_name": (self.exp_logger.run_name if self.exp_logger is not None else ""),
        }
        pm = ExperimentManagerClient(
            items_per_epoch=self.train_set.n_batches,
            epochs=self.n_epochs,
            show_recent_steps=self.progress_show_recent_steps,
            refresh_interval=self.progress_refresh_interval,
            custom_fields=pm_log_tags,
            host=self.progress_host,
            port=self.progress_port,
            dataset_name=run_meta["dataset_name"],
            model_name=run_meta["model_name"],
            run_name=run_meta["run_name"],
            run_dir=run_dir,
        )
        pm.mark_learning_rate_applied(self.model.lr)
        tm = None
        if self.enable_tensorboard:
            tm = TensorBoardManager(self.log_dir, tags=tm_log_tags, value_types=["scalar"] * len(tm_log_tags))
            tm.writer.add_text("Comments", "Training Started")
            tm.register("Visualization", "figure")
        ma_losses = {name: MovingAvg(self.moving_avg) for name in self.model.train_loss_names}

        best_loss = float('inf')

        if self.exp_logger is not None:
            self.exp_logger.set_status(STATUS_TRAINING)

        try:
            for epoch in range(self.n_epochs):
                loader = MultiThreadLoader(self.train_set, 3)
                for i, data_dict in enumerate(loader):
                    requested_lr = pm.consume_learning_rate_request()
                    if requested_lr is not None:
                        self.model.optimizer.param_groups[0]["lr"] = requested_lr
                        pm.mark_learning_rate_applied(self.model.lr)

                    # forward, backward, optimization
                    loss_dict, output_dict = self.model.trainStep(data_dict)

                    # Compute moving average of losses
                    for loss_name in self.model.train_loss_names:
                        ma_losses[loss_name].update(loss_dict[loss_name])
                        loss_dict[loss_name] = ma_losses[loss_name].get()

                    # Update progress manager
                    pm.update(epoch, i, LR=self.model.lr, **loss_dict)

                # Update tensorboard
                if tm is not None:
                    tm.log(pm.overall_progress, LR=self.model.lr, **loss_dict)

                if self.exp_logger is not None:
                    self.exp_logger.log_scalars(pm.overall_progress, LR=self.model.lr, **loss_dict)

                # Update learning rate scheduler
                self.lr_scheduler.update(loss_dict["Train/Main"])
                pm.mark_learning_rate_applied(self.model.lr)

                if epoch % self.eval_interval == 0:
                    if self.exp_logger is not None:
                        self.exp_logger.set_status(STATUS_EVALUATING)

                    eval_losses = self.evaluate(self.eval_set, pm=pm, tm=tm)

                    # Update tensorboard
                    if tm is not None:
                        tm.log(pm.overall_progress, **eval_losses)

                    if self.exp_logger is not None:
                        self.exp_logger.log_scalars(pm.overall_progress, **eval_losses)
                        self.exp_logger.set_status(STATUS_TRAINING)

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

            if self.exp_logger is not None:
                self.exp_logger.close(final_status=STATUS_DONE)
        except BaseException as e:
            if self.exp_logger is not None:
                self.exp_logger.close_with_error(str(e))
            raise
        finally:
            pm.close()


    def evaluate(self,
                 dataset: JimmyDataset,
                 compute_avg: bool=True,
                 pm: Optional[ExperimentManagerClient] = None,
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
        if "fig" in output_dict:
            fig = output_dict["fig"]
            if tm is not None:
                tm.log(pm.overall_progress, Visualization=fig)
            if self.exp_logger is not None and pm is not None:
                self.exp_logger.log_figure("Visualization", pm.overall_progress, fig)
            plt.close(fig)

        self.model.train()

        if compute_avg:     # if compute_avg, then average each loss over all batches
            return {name: torch.mean(eval_losses[name]).item() for name in self.model.eval_loss_names}

        return {name: eval_losses[name].cpu().numpy() for name in self.model.eval_loss_names}