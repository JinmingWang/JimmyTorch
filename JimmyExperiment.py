from JimmyTrainer import JimmyTrainer
from Datasets import *
from Models import *
from Training import *
import torch
from typing import *
from DynamicConfig import DynamicConfig
from datetime import datetime
import os
from rich import print as rprint
import pandas as pd
import yaml


class JimmyExperiment:
    """
    This is an example of an experiment class that defines the hyperparameters and constants for the experiment.
    For other type of experiments, or your customized trainer, you should write a new experiment class to accommodate
    the new set of hyperparameters and constants.
    """

    def __init__(self, comments: str, dir_name: str = None):
        self.comments = comments
        self.dir_name = dir_name if dir_name is not None else datetime.now().strftime("%y%m%d_%H%M%S")

        self.model_cfg = DynamicConfig(SampleCNN,
                                       optimizer_cls=torch.optim.Adam,
                                       optimizer_args={"lr": 1e-5},
                                       mixed_precision=False,
                                       compile_model=False,
                                       clip_grad=0.0)


        self.dataset_cfg = DynamicConfig(MNISTSampleDataset,
                                        batch_size=64,
                                        drop_last=False,
                                        shuffle=True)

        # The default hyperparameters for the experiment.
        self.lr_scheduler_cfg = DynamicConfig(torch.optim.lr_scheduler.ReduceLROnPlateau,
                                            mode='min',
                                            factor=0.5,
                                            threshold=1e-7,
                                            patience=10)

        # Other constants for the experiment.
        self.constants = {
            "n_epochs": 100,
            "moving_avg": 100,
            "eval_interval": 1,
            "save_dir": None,
            "log_dir": None,
        }

        # Allow trainer type selection for different training paradigms
        self.trainer_type = JimmyTrainer


    def __str__(self):
        return (f"Experiment{{\n"
                f"\tdataset={self.dataset_cfg}\n"
                f"\tmodel={self.model_cfg}\n"
                f"\tlr_scheduler={self.lr_scheduler_cfg}\n"
                f"\tconstants={self.constants}\n}}")


    def __repr__(self):
        return self.__str__()


    def start(self, checkpoint: str = None) -> JimmyTrainer:
        """
        Start the experiment with the given comments.
        :param checkpoint: The checkpoint to load the model from.
        :return: A `JimmyTrainer` object with almost everything during a training session.
        """
        rprint(f"[#00ff00]--- Start Experiment \"{self.comments}\" ---[/#00ff00]")

        self.dataset_cfg.set_name = "train"
        train_set = self.dataset_cfg.build()
        self.dataset_cfg.set_name = "eval"
        eval_set = self.dataset_cfg.build()

        model = self.model_cfg.build().to(DEVICE)

        if checkpoint is not None:
            model.loadFrom(checkpoint)

        model.initialize()
        self.lr_scheduler_cfg.optimizer = model.optimizer
        lr_scheduler = self.lr_scheduler_cfg.build()

        trainer_kwargs = {"train_set": train_set, "eval_set": eval_set, "model": model, "lr_scheduler": lr_scheduler}
        trainer_kwargs.update(self.constants)

        # Create Experiment directories with custom or auto-generated name
        if self.constants["save_dir"] is None:
            dataset_name = train_set.__class__.__name__
            model_name = model.__class__.__name__
            save_dir = f"Runs/{dataset_name}/{model_name}/{self.dir_name}/"
            self.constants["save_dir"] = save_dir
        else:
            self.constants["save_dir"] = os.path.join(self.constants["save_dir"], self.dir_name)

        if self.constants["log_dir"] is None:
            self.constants["log_dir"] = self.constants["save_dir"]
        else:
            self.constants["log_dir"] = os.path.join(self.constants["log_dir"], self.dir_name)

        # Create directories if they do not exist
        if not os.path.exists(self.constants['save_dir']):
            os.makedirs(self.constants['save_dir'])

        if not os.path.exists(self.constants['log_dir']):
            os.makedirs(self.constants['log_dir'])

        with open(os.path.join(self.constants['log_dir'], "model_arch.txt"), "w") as f:
            f.write(str(model))

        with open(os.path.join(self.constants['log_dir'], "comments.txt"), "w") as f:
            f.write(f"{self.comments}\n{self.__str__()}")

        rprint(f"[blue]Save directory: {self.constants['save_dir']}.[/blue]")
        rprint(f"[blue]Log directory: {self.constants['log_dir']}.[/blue]")

        trainer_kwargs["log_dir"] = self.constants['log_dir']
        trainer_kwargs["save_dir"] = self.constants['save_dir']

        trainer = self.trainer_type(**trainer_kwargs)
        trainer.start()

        return trainer


    def test(self, model, test_set) -> pd.DataFrame:
        """
        Test the model on a test set and return a detailed report.
        :param model: The trained model.
        :param test_set: The test dataset.
        :return: A pandas DataFrame with test results.
        """
        rprint(f"[blue]Testing on {self.comments}[/blue]")
        model.eval()

        test_losses = {name: torch.zeros(test_set.n_batches).to(DEVICE) for name in model.eval_loss_names}
        for i, data_dict in enumerate(test_set):
            loss_dict, output_dict = model.testStep(data_dict)

            for name in model.eval_loss_names:
                test_losses[name][i] = loss_dict[name]

        test_report = pd.DataFrame.from_dict({name: test_losses[name].cpu().numpy() for name in model.eval_loss_names})
        return test_report


