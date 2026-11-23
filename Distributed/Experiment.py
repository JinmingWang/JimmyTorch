from .Trainer import DistributedJimmyTrainer
from Training.ExperimentManager import ExperimentLogger, DEFAULT_SERVER_PORT
import torch
from DynamicConfig import DynamicConfig
from datetime import datetime
import os
from rich import print as rprint
import pandas as pd
from .Context import DistributedContext, moveToDevice
from .Model import DistributedJimmyModel
from .Dataset import DistributedJimmyDataset
from .Sample import DistributedSampleCNN, DistributedMNISTDataset
import uuid


class DistributedJimmyExperiment:
    """
    Own the opt-in distributed lifecycle, including rank-zero run artifacts.
    Configure the four DistributedJimmy classes without changing legacy projects.
    """

    def __init__(self, comments: str, dir_name: str = None):
        self.comments = comments
        self.dir_name = dir_name if dir_name is not None else datetime.now().strftime("%y%m%d_%H%M%S")

        self.model_cfg = DynamicConfig(DistributedSampleCNN,
                                       optimizer_cls=torch.optim.Adam,
                                       optimizer_args={"lr": 1e-5},
                                       mixed_precision=False,
                                       compile_model=False,
                                       clip_grad=0.0)


        self.dataset_cfg = DynamicConfig(DistributedMNISTDataset,
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
            "experiment_manager_server_port": DEFAULT_SERVER_PORT,
            "enable_tensorboard": False,
        }

        # Allow trainer type selection for different training paradigms
        self.trainer_type = DistributedJimmyTrainer
        self.distributed_cfg = {"backend": None, "timeout_seconds": 1800, "seed": 0,
                                "data_prepare_scope": "none"}


    def __str__(self):
        return (f"Experiment{{\n"
                f"\tdataset={self.dataset_cfg}\n"
                f"\tmodel={self.model_cfg}\n"
                f"\tlr_scheduler={self.lr_scheduler_cfg}\n"
                f"\tconstants={self.constants}\n}}")


    def __repr__(self):
        return self.__str__()


    def start(self, checkpoint: str = None) -> DistributedJimmyTrainer:
        """Own a fixed-membership torchrun lifecycle across one or many machines."""
        context = DistributedContext(backend=self.distributed_cfg["backend"],
                                     timeout_seconds=self.distributed_cfg["timeout_seconds"])
        exp_logger = None
        trainer_started = False
        try:
            self.dir_name = context.broadcastObject(self.dir_name)
            if context.is_primary:
                rprint(f'[green]--- Start Experiment "{self.comments}" ---[/green]')
            run_id = context.broadcastObject(uuid.uuid4().hex)
            topology = context.deviceAssignments()
            if any(entry["local_world_size"] != context.local_world_size for entry in topology):
                raise ValueError("All machines must use the same local worker count")
            config_signatures = context.gatherObjects((str(self.model_cfg), str(self.dataset_cfg),
                str(self.lr_scheduler_cfg), self.constants, self.distributed_cfg))
            if any(signature != config_signatures[0] for signature in config_signatures):
                raise ValueError("Experiment configurations must match on every rank")
            if not issubclass(self.model_cfg.cls, DistributedJimmyModel):
                raise TypeError("model_cfg.cls must inherit DistributedJimmyModel")
            if not issubclass(self.dataset_cfg.cls, DistributedJimmyDataset):
                raise TypeError("dataset_cfg.cls must inherit DistributedJimmyDataset")
            dataset_args = {key: value for key, value in vars(self.dataset_cfg).items()
                            if key != "cls" and key != "set_name" and not key.startswith("_")}
            prepare_scope = self.distributed_cfg["data_prepare_scope"]
            if prepare_scope not in ("none", "shared", "per_node"):
                raise ValueError("data_prepare_scope must be none, shared, or per_node")
            if prepare_scope == "shared":
                context.runPrimary(lambda: self.dataset_cfg.cls.prepareData(**dataset_args))
            elif prepare_scope == "per_node":
                context.runAll(lambda: self.dataset_cfg.cls.prepareData(**dataset_args)
                               if context.is_node_primary else None)
            torch.manual_seed(self.distributed_cfg["seed"])
            train_set = context.runAll(lambda: self.dataset_cfg.cls(set_name="train", **dataset_args))
            eval_set = None
            def buildValidation():
                nonlocal eval_set
                eval_set = self.dataset_cfg.cls(set_name="eval", **dataset_args)
            context.runPrimary(buildValidation)
            model = context.runAll(lambda: self.model_cfg.build().to(context.device))
            if checkpoint is not None:
                context.runPrimary(lambda: model.loadFrom(checkpoint))
            context.runAll(model.initialize)
            self.lr_scheduler_cfg.optimizer = model.optimizer
            lr_scheduler = context.runAll(self.lr_scheduler_cfg.build)
            dataset_name = train_set.__class__.__name__
            model_name = model.__class__.__name__
            save_base = self.constants["save_dir"]
            log_base = self.constants["log_dir"]
            save_dir = os.path.join(save_base or f"Runs/{dataset_name}/{model_name}", self.dir_name)
            log_dir = os.path.join(log_base, self.dir_name) if log_base else save_dir
            save_dir, log_dir = context.broadcastObject((save_dir, log_dir))

            def prepareRun():
                nonlocal exp_logger
                os.makedirs(save_dir, exist_ok=True)
                os.makedirs(log_dir, exist_ok=True)
                with open(os.path.join(log_dir, "model_arch.txt"), "w") as output_file:
                    output_file.write(str(model))
                with open(os.path.join(log_dir, "comments.txt"), "w") as output_file:
                    output_file.write(self.comments)
                exp_logger = ExperimentLogger(log_dir, dataset_name, model_name, self.dir_name)
                exp_logger.set_comments(self.comments)
                exp_logger.set_model_arch(str(model))
                exp_logger.log_hparams({
                    "dataset": str(self.dataset_cfg), "model": str(self.model_cfg),
                    "lr_scheduler": str(self.lr_scheduler_cfg),
                    **{key: value for key, value in self.constants.items() if key not in ("save_dir", "log_dir")},
                    "world_size": context.world_size, "distributed": str(self.distributed_cfg), "batch_size_per_gpu": train_set.batch_size,
                    "global_batch_size": train_set.batch_size * context.world_size,
                })
            context.runPrimary(prepareRun)
            trainer_kwargs = dict(self.constants)
            trainer_kwargs.update(train_set=train_set, eval_set=eval_set, model=model,
                                  lr_scheduler=lr_scheduler, save_dir=save_dir, log_dir=log_dir,
                                  exp_logger=exp_logger)
            trainer_kwargs.update(distributed_context=context,
                                  distributed_seed=self.distributed_cfg["seed"], run_id=run_id, topology=topology)
            trainer_kwargs["progress_port"] = trainer_kwargs.pop("experiment_manager_server_port")
            trainer = context.runAll(lambda: self.trainer_type(**trainer_kwargs))
            trainer_started = True
            trainer.start()
            return trainer
        except BaseException as error:
            if exp_logger is not None and not trainer_started:
                exp_logger.close_with_error(str(error))
            raise
        finally:
            context.close()


    def test(self, model, test_set) -> pd.DataFrame:
        """
        Test the model on a test set and return a detailed report.
        :param model: The trained model.
        :param test_set: The test dataset.
        :return: A pandas DataFrame with test results.
        """
        rprint(f"[blue]Testing on {self.comments}[/blue]")
        model.eval()

        test_losses = {name: torch.zeros(test_set.n_batches) for name in model.eval_loss_names}
        for i, data_dict in enumerate(test_set):
            loss_dict, output_dict = model.testStep(moveToDevice(data_dict, next(model.parameters()).device))

            for name in model.eval_loss_names:
                test_losses[name][i] = loss_dict[name]

        test_report = pd.DataFrame.from_dict({name: test_losses[name].cpu().numpy() for name in model.eval_loss_names})
        return test_report


