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


class JimmyExperiment:
    """
    This is an example of an experiment class that defines the hyperparameters and constants for the experiment.
    For other type of experiments, or your customized trainer, you should write a new experiment class to accommodate
    the new set of hyperparameters and constants.
    """

    def __init__(self, comments: str):
        self.comments = comments

        self.model_cfg = DynamicConfig(SampleCNN,
                                       optimizer_cls=torch.optim.Adam,
                                       optimizer_args={"lr": 1e-5},
                                       mixed_precision=False,
                                       clip_grad=0.0)


        self.dataset_cfg = DynamicConfig(MNISTSampleDataset,
                                        batch_size=64,
                                        drop_last=False,
                                        shuffle=True)

        # The default hyperparameters for the experiment.
        self.lr_scheduler_cfg = DynamicConfig(JimmyLRScheduler,
                                              peak_lr=2e-4,
                                              min_lr=1e-7,
                                              warmup_count=10,
                                              window_size=10,
                                              patience=10,
                                              decay_rate=0.5)

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


    def start(self, checkpoint: str = None) -> JimmyTrainer:
        """
        Start the experiment with the given comments.
        :param checkpoint: The checkpoint to load the model from.
        :return: A `JimmyTrainer` object with amost everything during a training session.
        """
        rprint(f"[#00ff00]--- Start Experiment \"{self.comments}\" ---[/#00ff00]")

        self.dataset_cfg.set_name = "train"
        train_set = self.dataset_cfg.build()
        self.dataset_cfg.set_name = "eval"
        eval_set = self.dataset_cfg.build()

        model = self.model_cfg.build().to(DEVICE)
        if self.constants["compile_model"]:
            model: JimmyModel = torch.compile(model)

        if checkpoint is not None:
            model.loadFrom(checkpoint)

        model.initOptimizer()
        self.lr_scheduler_cfg.optimizer = model.optimizer
        lr_scheduler = self.lr_scheduler_cfg.build()

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
        self.dataset_cfg.set_name = "test"
        test_set = self.dataset_cfg.build()
        test_losses = trainer.evaluate(test_set, compute_avg=False)

        test_report = pd.DataFrame.from_dict(test_losses)
        test_report.to_csv(os.path.join(log_dir, "test_report.csv"))

        rprint(f"[blue]Testing done. Reports saved to: {os.path.join(log_dir, 'test_report.csv')}.[/blue]")

        return trainer


