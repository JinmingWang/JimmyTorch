from JimmyTrainer import JimmyTrainer
from Datasets import *
from Models import *
from Training import *
import torch
from typing import *
from copy import deepcopy


class JimmyExperiment:
    """
    This is an example of an experiment class that defines the hyperparameters and constants for the experiment.
    For other type of experiments, or your customized trainer, you should write a new experiment class to accommodate
    the new set of hyperparameters and constants.
    """
    instance_keys = ["dataset", "model", "optimizer", "lr_scheduler"]

    def __init__(self):
        # The default hyperparameters for the experiment.
        self.hyper_params: dict[str, Any] = {
            "dataset": MNISTDataset,
            "dataset_args": {"set_name": "train", "batch_size": 64, "drop_last": False, "shuffle": True},

            "model": SampleCNN,
            "model_args": {},

            "optimizer": torch.optim.Adam,
            "optimizer_args": {"lr": 1e-5},

            "lr_scheduler": JimmyLRScheduler,
            "lr_scheduler_args": {
                "init_lr": 1e-5,
                "peak_lr": 2e-4,
                "min_lr": 1e-7,
                "warmup_count": 10,
                "window_size": 10,
                "patience": 10,
                "decay_rate": 0.5
            },
        }

        # Other constants for the experiment.
        self.constants = {
            "n_epochs": 100,
            "moving_avg": 100,
            "mixed_precision": False,
            "compile_model": False,
            "clip_grad": 0.0,
        }


    def __str__(self):
        return f"Experiment with:\n\thyper_params={self.hyper_params}\n\tconstants={self.constants}"


    def __repr__(self):
        return self.__str__()


    def start(self, comments: str = "") -> JimmyTrainer:
        """
        Start the experiment with the given comments.
        :param comments: Comments to be added to the Experiment.
        :return: A `JimmyTrainer` object with amost everything during a training session.
        """
        rich_comments = f"{comments}.\n{self.__str__()}"
        trainer_kwargs = {k: self.hyper_params[k](**self.hyper_params[f"{k}_args"]) for k in self.instance_keys}
        trainer_kwargs.update(self.constants)

        trainer = JimmyTrainer(**trainer_kwargs, comments=rich_comments)
        return trainer


