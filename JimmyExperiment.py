from JimmyTrainer import JimmyTrainer
from Datasets import *
from Models import *
from Training import *
import torch
from typing import *
from datetime import datetime
import os
from rich import print as rprint
import pandas as pd


class JimmyExperiment:
    """
    This is an example of an experiment class that defines the hyperparameters and constants for the experiment.
    For other type of experiments, or your customized trainer, you should write a new experiment class to accommodate
    the new set of hyperparameters and constants.
    """

    def __init__(self, comments: str):
        self.comments = comments

        self.model_cfg: dict[str, Any] = {
            "class": SampleCNN,
            "args": {
                "optimizer_cls": torch.optim.Adam,
                "optimizer_args": {"lr": 1e-5},
                "mixed_precision": False,
                "clip_grad": 0.0,
            }
        }

        self.dataset_cfg: dict[str, Any] = {
            "class": MNISTSampleDataset,
            "args": {
                "batch_size": 64,
                "drop_last": False,
                "shuffle": True}
        }

        # The default hyperparameters for the experiment.
        self.lr_scheduler_cfg: dict[str, Any] = {
            "class": JimmyLRScheduler,
            "args": {
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
            "compile_model": False,
        }


    def __str__(self):
        return (f"Experiment{{\n"
                f"\tdataset={self.dataset_cfg}\n"
                f"\tmodel={self.model_cfg}\n"
                f"\tlr_scheduler={self.lr_scheduler_cfg}\n"
                f"\tconstants={self.constants}\n}}")


    def __repr__(self):
        return self.__str__()


    def start(self) -> JimmyTrainer:
        """
        Start the experiment with the given comments.
        :param comments: Comments to be added to the Experiment.
        :return: A `JimmyTrainer` object with amost everything during a training session.
        """
        rprint(f"[#00ff00]--- Start Experiment \"{self.comments}\" ---[/#00ff00]")

        train_set = self.dataset_cfg["class"](set_name="train", **self.dataset_cfg["args"])
        eval_set = self.dataset_cfg["class"](set_name="eval", **self.dataset_cfg["args"])
        model = self.model_cfg["class"](**self.model_cfg["args"]).to(DEVICE)
        model.initOptimizer()
        lr_scheduler = self.lr_scheduler_cfg["class"](model.optimizer, **self.lr_scheduler_cfg["args"])

        trainer_kwargs = {"train_set": train_set, "eval_set": eval_set, "model": model, "lr_scheduler": lr_scheduler}
        trainer_kwargs.update(self.constants)

        # Create Experiment directories
        now_str = datetime.now().strftime("%y%m%d_%H%M%S")
        dataset_name = trainer_kwargs["train_set"].__class__.__name__
        model_name = model.__class__.__name__
        save_dir = f"Runs/{dataset_name}/{model_name}/{now_str}/"
        log_dir = save_dir

        # Create directories if they do not exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        with open(os.path.join(log_dir, "model_arch.txt"), "w") as f:
            f.write(str(model))

        with open(os.path.join(log_dir, "comments.txt"), "w") as f:
            f.write(f"{self.comments}.\n{self.__str__()}")

        rprint(f"[blue]Save directory: {save_dir}.[/blue]")
        rprint(f"[blue]Log directory: {log_dir}.[/blue]")

        trainer_kwargs["log_dir"] = log_dir
        trainer_kwargs["save_dir"] = save_dir

        trainer = JimmyTrainer(**trainer_kwargs)
        trainer.start()

        rprint(f"[blue]Training done. Start testing.[/blue]")
        test_set = self.dataset_cfg["class"](set_name="test", **self.dataset_cfg["args"])
        test_losses = trainer.evaluate(test_set, compute_avg=False)

        test_report = pd.DataFrame.from_dict(test_losses)
        test_report.to_csv(os.path.join(log_dir, "test_report.csv"))

        rprint(f"[blue]Testing done. Reports saved to: {os.path.join(log_dir, 'test_report.csv')}.[/blue]")

        return trainer


