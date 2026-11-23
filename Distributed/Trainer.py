import torch
from .Dataset import DistributedJimmyDataset
from .Loader import DistributedLoader
from .MovingAverage import MovingAvg
from Training.ExperimentManager import (
    ExperimentLogger,
    STATUS_DONE,
    STATUS_ERROR,
    STATUS_EVALUATING,
    STATUS_TRAINING,
)
from .Model import DistributedJimmyModel
import matplotlib.pyplot as plt
import os
from typing import Any, Optional
import uuid
from .Context import DistributedContext, moveToDevice
from .Reporting import DistributedRunClient, NodeReporter


class DistributedJimmyTrainer:
    def __init__(self,
                 train_set: DistributedJimmyDataset,
                 eval_set: DistributedJimmyDataset,
                 model: DistributedJimmyModel,
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
                 enable_tensorboard: bool = False,
                 distributed_context: Optional[DistributedContext] = None,
                 distributed_seed: int = 0, run_id: str = "", topology=None) -> None:
        """
        Initialize the trainer with a dataset, model, optimizer, and comments.

        :param train_set: A `JimmyDataset` object that provides the training data. It must return a dictionary containing the input data and target labels.
        :param model: An initialized `DistributedJimmyModel` with conventional trainStep/backwardOptimize.
        :param lr_scheduler: A learning rate scheduler. It can be a `JimmyLRScheduler` or a PyTorch learning rate scheduler. The scheduler must implement an `update` method that takes the current loss as an argument.
        :param log_dir: A string specifying the directory where the training logs will be saved.
        :param save_dir: A string specifying the directory where the model checkpoints will be saved.
        :param moving_avg: An integer specifying the window size for calculating the moving average of the loss. Default is 100.
        :param eval_interval: An integer specifying the interval (in epochs) at which to evaluate the model on the validation set.
        :param early_stop_lr: A float specifying the learning rate threshold for early stopping. Default is 0.0 (no early stopping).

        :param progress_show_recent_steps: show recent steps in GUI
        :param progress_refresh_interval: refresh interval for GUI (in seconds)
        """

        if n_epochs < 1 or eval_interval < 1 or moving_avg < 1:
            raise ValueError("n_epochs, eval_interval, and moving_avg must be positive.")
        self.distributed_seed = distributed_seed
        self.run_id = run_id
        self.topology = topology
        self.distributed_context = distributed_context
        self.train_set = train_set
        self.eval_set = eval_set
        self.model = model

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
        owns_context = self.distributed_context is None
        context = self.distributed_context or DistributedContext()
        self.distributed_context = context
        self.device = context.device
        pm, tm, loader, reporter = None, None, None, None
        logger_closed = False
        try:
            def validateSetup():
                if not isinstance(self.model, DistributedJimmyModel):
                    raise TypeError("Use DistributedJimmyModel for distributed training")
                if not isinstance(self.train_set, DistributedJimmyDataset):
                    raise TypeError("Use DistributedJimmyDataset for distributed training")
                self.train_set.configureDistributed(context.rank, context.world_size, self.distributed_seed)
                if self.train_set.n_batches == 0:
                    raise ValueError("Training dataset produces zero batches per rank.")
                if context.is_primary and (self.eval_set is None or self.eval_set.n_batches == 0):
                    raise ValueError("Validation dataset produces zero batches.")
            context.runAll(validateSetup)
            contracts = context.gatherObjects((self.train_set.n_samples, self.train_set.batch_size,
                self.train_set.n_batches, self.train_set.drop_last, self.train_set.shuffle,
                self.model.train_loss_names, self.model.eval_loss_names, self.n_epochs, self.eval_interval))
            if any(contract != contracts[0] for contract in contracts):
                raise ValueError("Dataset, metric and training configuration must match on every rank")
            self.model.enableDistributed(context)
            device_assignments = self.topology or context.deviceAssignments()
            self.run_id = context.broadcastObject(self.run_id or uuid.uuid4().hex)
            node_assignments = [entry for entry in device_assignments if entry["node_id"] == context.node_id]
            def setupReporting():
                nonlocal pm, tm
                run_meta = self.exp_logger
                pm = DistributedRunClient(
                    run_id=self.run_id, topology=device_assignments,
                    items_per_epoch=self.train_set.n_batches, epochs=self.n_epochs,
                    show_recent_steps=self.progress_show_recent_steps,
                    refresh_interval=self.progress_refresh_interval,
                    custom_fields=self.model.train_loss_names + ["LR"],
                    host=self.progress_host, port=self.progress_port,
                    dataset_name=run_meta.dataset_name if run_meta else "",
                    model_name=run_meta.model_name if run_meta else "",
                    run_name=run_meta.run_name if run_meta else "",
                    run_dir=run_meta.run_dir if run_meta else self.save_dir,
                    device_assignments=node_assignments,
                )
                pm.mark_learning_rate_applied(self.model.lr)
                if self.enable_tensorboard:
                    from Training.TensorBoardManager import TensorBoardManager
                    tags = self.model.train_loss_names + self.model.eval_loss_names + ["LR"]
                    tm = TensorBoardManager(self.log_dir, tags=tags, value_types=["scalar"] * len(tags))
                    tm.register("Visualization", "figure")
                if self.exp_logger:
                    self.exp_logger.set_status(STATUS_TRAINING)
            context.runPrimary(setupReporting)
            if context.is_node_primary:
                reporter = NodeReporter(self.progress_host, self.progress_port, self.run_id,
                    context.node_id, node_assignments, self.progress_refresh_interval)
            ma_losses = {name: MovingAvg(self.moving_avg) for name in self.model.train_loss_names}
            best_loss = float("inf")
            self.model.train()
            self.model.optimizer.zero_grad()
            for epoch in range(self.n_epochs):
                self.train_set.setEpoch(epoch)
                loader = DistributedLoader(self.train_set, 3, device=context.device)
                try:
                    batch_iterator = iter(loader)
                    for batch_idx in range(self.train_set.n_batches):
                        data_dict = context.runAll(lambda: next(batch_iterator))
                        requested_lr = context.broadcastObject(pm.consume_learning_rate_request() if pm else None)
                        if requested_lr is not None:
                            self.model.optimizer.param_groups[0]["lr"] = requested_lr
                        data_dict = moveToDevice(data_dict, context.device)
                        self.model.beginStep()
                        loss_dict, output_dict = self.model.trainStep(data_dict)
                        context.runAll(self.model.endStep)
                        def validateMetrics():
                            return {name: float(loss_dict[name]) for name in self.model.train_loss_names}
                        loss_dict = context.runAll(validateMetrics)
                        loss_dict = context.averageMetrics(loss_dict, self.model.train_loss_names)
                        for loss_name in self.model.train_loss_names:
                            ma_losses[loss_name].update(loss_dict[loss_name])
                            loss_dict[loss_name] = ma_losses[loss_name].get()
                        if pm:
                            pm.mark_learning_rate_applied(self.model.lr)
                            pm.update(epoch, batch_idx, LR=self.model.lr, **loss_dict)
                finally:
                    loader.close()
                    loader = None

                def logEpoch():
                    if tm:
                        tm.log(pm.overall_progress, LR=self.model.lr, **loss_dict)
                    if self.exp_logger:
                        self.exp_logger.log_scalars(pm.overall_progress, LR=self.model.lr, **loss_dict)
                context.runPrimary(logEpoch)
                if hasattr(self.lr_scheduler, "update"):
                    self.lr_scheduler.update(loss_dict["Train/Main"])
                elif isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(loss_dict["Train/Main"])
                else:
                    self.lr_scheduler.step()
                if pm:
                    pm.mark_learning_rate_applied(self.model.lr)

                if epoch % self.eval_interval == 0:
                    def evaluatePrimary():
                        if self.exp_logger:
                            self.exp_logger.set_status(STATUS_EVALUATING)
                        if pm:
                            pm.set_status(STATUS_EVALUATING)
                        eval_losses = self.evaluate(self.eval_set, pm=pm, tm=tm)
                        if tm:
                            tm.log(pm.overall_progress, **eval_losses)
                        if self.exp_logger:
                            self.exp_logger.log_scalars(pm.overall_progress, **eval_losses)
                            self.exp_logger.set_status(STATUS_TRAINING)
                        if pm:
                            pm.set_status(STATUS_TRAINING)
                        return eval_losses
                    eval_losses = context.runPrimary(evaluatePrimary)
                    eval_loss = eval_losses["Eval/Main"]
                    def saveCheckpoints():
                        os.makedirs(self.save_dir, exist_ok=True)
                        if eval_loss < best_loss:
                            self.model.saveTo(os.path.join(self.save_dir, "best.pth"))
                        self.model.saveTo(os.path.join(self.save_dir, "last.pth"))
                    context.runPrimary(saveCheckpoints)
                    best_loss = min(best_loss, eval_loss)
                should_stop = context.broadcastObject(
                    self.early_stop_lr > 0 and self.model.lr < self.early_stop_lr)
                if should_stop:
                    break
            if context.is_primary:
                if pm:
                    pm.set_status(STATUS_DONE)
                if self.exp_logger:
                    self.exp_logger.close(final_status=STATUS_DONE)
                    logger_closed = True
        except BaseException as error:
            if pm:
                pm.set_status(STATUS_ERROR)
            if context.is_primary and self.exp_logger and not logger_closed:
                self.exp_logger.close_with_error(str(error))
            raise
        finally:
            if loader:
                loader.close()
            if isinstance(self.model, DistributedJimmyModel):
                self.model.disableDistributed()
            if reporter:
                reporter.close()
            if pm:
                pm.close()
            if tm:
                tm.writer.close()
            if owns_context:
                context.close()

    def evaluate(self, dataset, compute_avg=True, pm=None, tm=None):
        """Evaluate locally; the training loop invokes this only on rank zero."""
        device = next(self.model.parameters()).device
        eval_losses = {name: [] for name in self.model.eval_loss_names}
        was_training = self.model.training
        self.model.eval()
        last_fig = None
        try:
            with torch.no_grad():
                for data_dict in dataset:
                    loss_dict, output_dict = self.model.evalStep(moveToDevice(data_dict, device))
                    for name in self.model.eval_loss_names:
                        eval_losses[name].append(float(loss_dict[name]))
                    if last_fig is not None:
                        plt.close(last_fig)
                    fig = output_dict.get("fig")
                    last_fig = fig if hasattr(fig, "savefig") else None
            if not eval_losses[self.model.eval_loss_names[0]]:
                raise ValueError("Cannot evaluate an empty dataset.")
            if last_fig is not None:
                step = pm.overall_progress if pm else 0
                if tm:
                    tm.log(step, Visualization=last_fig)
                if self.exp_logger:
                    self.exp_logger.log_figure("Visualization", step, last_fig)
        finally:
            if last_fig is not None:
                plt.close(last_fig)
            self.model.train(was_training)
        if compute_avg:
            return {name: sum(values) / len(values) for name, values in eval_losses.items()}
        import numpy as np
        return {name: np.asarray(values) for name, values in eval_losses.items()}
