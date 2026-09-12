# Project architecture and source navigation

Read this reference when locating JimmyTorch inside a consuming project or deciding where a change belongs.

## Framework boundaries

The framework source is anchored by:

- `Datasets/`: base batching contract and specialized preprocessing.
- `Models/`: base model contract, layers, losses, attention, and cost helpers.
- `DiffusionModels/`: DDPM and DDIM processes.
- `Training/`: moving averages, optional TensorBoard support, and Experiment Manager package.
- `DynamicConfig.py`: deferred construction.
- `JimmyExperiment.py`: experiment assembly and directory creation.
- `JimmyTrainer.py`: default loop, scheduling, evaluation, checkpoints, and live reporting.

A consuming research project commonly adds its own `Datasets/`, `Models/`, experiment/trainer subclasses, `main_*.py` entrypoints, and report code around an embedded `JimmyTorch/` checkout. Infer the real layout from imports and repository files rather than enforcing a template blindly.

## Runtime data flow

1. A dataset yields a named batch dictionary.
2. A model phase step consumes that dictionary and returns named losses plus outputs.
3. The trainer smooths/logs metrics, updates the scheduler, evaluates, and saves checkpoints.
4. The experiment constructs all components and owns the run directory and logger.
5. The local SQLite record is durable; the optional live client reports to the standalone dashboard service.

Interfaces between these stages are the highest-risk change points. When renaming a dictionary or loss key, search all producers, consumers, logging tags, and tests.

## Generated run layout

The default path is `Runs/{DatasetClass}/{ModelClass}/{run_name}/`. A current run may contain:

- `best.pth` and `last.pth`
- `comments.txt` and `model_arch.txt`
- `status_and_log.sqlite`
- TensorBoard event files only when enabled
- Project-generated evaluation reports or other artifacts

`Runs/Experiment_GUI_Status.sqlite` stores dashboard-wide settings. Treat all of `Runs/` as generated/user data: inspect when useful, but do not rewrite, delete, or use it as a source fixture without explicit scope.

## Change placement

- Project-specific data semantics belong in project datasets.
- Network architecture and task loss semantics belong in project models/components.
- Reusable tensor operations belong in framework utilities only when genuinely general.
- Alternate training paradigms belong in custom trainer/experiment classes unless the base behavior itself is in scope.
- Dashboard behavior spans the Python server/storage/client and React frontend; trace both before editing a wire shape.
