---
name: jimmytorch
description: Work safely and consistently in projects that use the JimmyTorch research framework. Use when implementing or changing JimmyDataset, JimmyModel, JimmyExperiment, JimmyTrainer, experiment monitoring, or JimmyTorch utilities; do not use for unrelated PyTorch projects.
---

# JimmyTorch

Use this skill to preserve JimmyTorch's runtime contracts while adapting it to a research project. Treat the checked-out source as authoritative: inspect relevant definitions before changing call sites or documentation.

## Locate the framework

JimmyTorch may be the repository root or a `JimmyTorch/` directory embedded in a larger project. Locate these anchors before acting:

- `JimmyExperiment.py` and `JimmyTrainer.py`
- `Datasets/JimmyDataset.py`
- `Models/JimmyModel.py`
- `Training/ExperimentManager/`

Distinguish framework code from the consuming project's datasets, models, experiment subclasses, entry scripts, reports, and generated `Runs/`. Do not assume framework code is read-only when the user explicitly asks to change JimmyTorch itself.

## Preserve the core contracts

- A `JimmyDataset` subclass calls the base initializer, sets `n_samples`, and returns a dictionary for each zero-based batch index.
- A `JimmyModel` exposes `train_loss_names` and `eval_loss_names`; step methods return `(loss_dict, output_dict)`.
- `Train/Main` drives the scheduler and `Eval/Main` selects `best.pth`; keep both keys consistent with declared names.
- Move a model to its device before `initialize()`. When resuming through `JimmyExperiment.start(checkpoint)`, model weights load before optimizer creation.
- Keep training independent from the optional Experiment Manager service. Durable run logging is local; live HTTP reporting must remain best-effort.
- Do not invent configuration fields, generated files, or public exports. Check the current constructors, package `__init__.py` files, and call sites.

## Route by task

Read only the references needed for the request:

- Project layout, ownership, or source discovery: [references/architecture.md](references/architecture.md)
- Dataset implementation or loading changes: [references/datasets.md](references/datasets.md)
- Model, loss, checkpoint, or reusable-layer changes: [references/models.md](references/models.md)
- Experiment configuration, training, evaluation, or resume behavior: [references/training.md](references/training.md)
- Dashboard, SQLite records, live reporting, or learning-rate control: [references/experiment-manager.md](references/experiment-manager.md)
- Diffusion, trajectory, sequence, cost, or 3D helpers: [references/utilities.md](references/utilities.md)
- Exact public-name lookup: [references/api.md](references/api.md)

When a task crosses boundaries, read the smallest combination that covers it. For example, a new experiment normally needs datasets, models, and training; a monitor-only fix needs the Experiment Manager reference and the relevant source files.

## Working method

1. Inspect the consuming project's existing dataset/model/experiment patterns and the current framework definitions.
2. Identify which dictionary keys and loss names form the interface between components.
3. Make the smallest change that preserves those interfaces, or update all consumers together when the interface must change.
4. Validate with CPU-safe unit or smoke tests when possible. Avoid downloading datasets or starting long training runs unless requested.
5. For monitor changes, test persistence and live reporting separately; a monitor failure must not become a training failure.

Generated run databases, checkpoints, figures, and frontend build outputs are evidence or artifacts, not source templates. Do not edit a real run or use destructive dashboard operations without explicit scope.
