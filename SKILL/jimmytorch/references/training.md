# Training and experiment contracts

Read this reference for `DynamicConfig`, `JimmyExperiment`, `JimmyTrainer`, custom trainers, resume logic, evaluation, or checkpoints.

## Configuration flow

`DynamicConfig(cls, **kwargs)` stores public attributes and builds `cls(**kwargs)`. Update it with attribute assignment or `add(**kwargs)`; remove a key with `remove(key)`.

`JimmyExperiment.start(checkpoint=None)`:

1. Builds train and eval datasets by changing `dataset_cfg.set_name`.
2. Builds the model, moves it to `Datasets.DEVICE`, optionally loads the checkpoint, and initializes its optimizer.
3. Supplies that optimizer to `lr_scheduler_cfg` and builds the scheduler.
4. Resolves run/log directories, writes readable comments and architecture files, and opens `ExperimentLogger`.
5. Builds `trainer_type`, runs it synchronously, and returns the trainer.

The default run path is `Runs/{DatasetClass}/{ModelClass}/{dir_name}/`; `dir_name` defaults to a timestamp. Supplied `save_dir` and `log_dir` act as bases to which the run name is appended.

## Default experiment constants

- `n_epochs=100`
- `moving_avg=100`
- `eval_interval=1`
- `save_dir=None`
- `log_dir=None`
- `experiment_manager_server_port=9000`
- `enable_tensorboard=False`

The trainer also accepts `early_stop_lr`, `progress_show_recent_steps`, `progress_refresh_interval`, and `progress_host` when added to the constants.

## Loop invariants

- Training uses `MultiThreadLoader(train_set, 3)` and smooths each declared training loss independently.
- The scheduler's `update` receives smoothed `Train/Main` after each epoch. A PyTorch scheduler without `update` is adapted from its `step` signature.
- Evaluation averages each declared loss over validation batches.
- On evaluation epochs, lower `Eval/Main` updates `best.pth`; `last.pth` is also written. It is therefore the latest **evaluation** checkpoint, not necessarily the latest epoch.
- A positive `early_stop_lr` stops training when current learning rate drops below the threshold.
- Logger status must close as done or error, and the live client must close in `finally`.

`JimmyExperiment.test(model, test_set)` calls `testStep` and returns a pandas DataFrame with one row per batch, not per sample.

## Custom trainers

Set `experiment.trainer_type` to a compatible class for another paradigm. Preserve the constructor inputs used by `JimmyExperiment`, run directory semantics, main loss keys, logger lifecycle, and best-effort monitor boundary.

## Validation

Use a tiny synthetic dataset and model to test component construction, one optimizer step, evaluation aggregation, scheduler update, checkpoint selection, resume, error closure, and operation with no monitor service. Avoid running the downloadable MNIST example unless network access and runtime cost are in scope.

## Separate distributed lifecycle

Use `DistributedJimmyExperiment`, `DistributedJimmyTrainer`, `DistributedJimmyDataset`, and `DistributedJimmyModel` from `Distributed`. Never retrofit the original pipeline for DDP. Launch `distributed_main.py` with torchrun, on one or multiple machines with equal worker counts. Batch size is per GPU; LR scaling is explicit.

The context selects the device before construction. Preparation scope is `none`, `shared`, or `per_node`; all ranks must see the same logical dataset/order. Global rank 0 owns validation, SQLite, figures, checkpoints and overall dashboard status. Every machine's local rank 0 reports telemetry without collectives. Weight-only checkpoints do not resume optimizer/scheduler state. Fixed membership only; no gradient accumulation, multiple optimizers, or model sharding.

Run `tests/test_distributed.py`, `tests/test_legacy_compatibility.py`, and `tests/test_gpu_monitoring.py`, plus frontend checks. Distinguish CPU/two-launcher simulation results from pending real multi-machine and CUDA verification. Full interfaces and launch recipes: `docs/distributed.html`.
