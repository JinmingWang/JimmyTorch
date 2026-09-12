<h1 align="center">
  <img src="docs/assets/images/jimmytorch-tensor-prism.svg" width="240" alt="JimmyTorch prism logo representing Datasets, Models, and Experiments">
</h1>

<p align="center">
  A compact PyTorch research framework for turning datasets, models, and training loops into reproducible, observable experiments.
</p>

<p align="center">
  <a href="docs/index.html"><img alt="Documentation" src="https://img.shields.io/badge/docs-HTML-54e0b1?style=flat-square"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-framework-EE4C2C?style=flat-square&logo=pytorch&logoColor=white">
  <a href="LICENSE"><img alt="Apache-2.0 license" src="https://img.shields.io/badge/license-Apache--2.0-f5c76a?style=flat-square"></a>
</p>

<p align="center">
  <a href="README.md">English</a> · <a href="README.cn.md">简体中文</a>
</p>

<p align="center">
  <img src="docs/assets/images/architecture.svg" alt="JimmyTorch architecture: datasets flow through models and training into durable experiment records">
</p>

## One loop, from research code to evidence

JimmyTorch provides a small set of explicit contracts rather than hiding research code behind a large training platform:

| Build | Run | Observe |
|---|---|---|
| Dictionary-based datasets and reusable model components | Configuration-as-code, mixed precision, evaluation, scheduling, and checkpoints | SQLite run records, live progress, comparisons, figures, notes, and model architecture |

The training loop remains independent from the browser service. Runs are logged locally, while live status reporting is asynchronous and best-effort.

## A combinatorial experiment space

<p align="center">
  <img src="docs/assets/images/experiment-space.svg" alt="A woven JimmyTorch experiment lattice: every dataset crosses every model, each pairing supports multiple runs, and flexible contours group them into experiments">
</p>

Datasets and models are reusable dimensions rather than one-to-one pairs. Combine them freely, repeat each pairing with different run configurations, and retain the results as comparable experiments.

## Experiment Manager

<p align="center">
  <img src="docs/assets/images/experiment-manager.png" alt="Sanitized JimmyTorch Experiment Manager dashboard showing live progress and multi-run comparison">
</p>

One standalone dashboard can follow current training and inspect historical runs. It includes resource metrics, runtime learning-rate control, a searchable run tree, multi-run scalar curves, stored figures, summaries, notes, and architecture views.

## Quick start

```bash
git clone https://github.com/JinmingWang/JimmyTorch.git
cd JimmyTorch

# Optional: start the dashboard in one terminal.
python -m Training.ExperimentManager

# Run the included MNIST example in another terminal.
python main.py
```

The repository is currently used directly from source and does not provide a pinned dependency manifest. Install PyTorch for your platform, then see the [getting-started guide](docs/getting-started.html) for runtime and optional dependencies.

## Structure

```text
Datasets/                     Batch contracts and preprocessing utilities
Models/                       Model base class, layers, losses, and attention
DiffusionModels/              DDPM and DDIM processes
Training/ExperimentManager/   Local logging, live client, server, and dashboard
JimmyExperiment.py            Component assembly and run lifecycle
JimmyTrainer.py               Training, evaluation, scheduling, and checkpoints
SKILL/jimmytorch/              LLM-facing project guidance
docs/                          Human-facing documentation
```

## Documentation

- [Architecture and overview](docs/index.html)
- [Getting started](docs/getting-started.html)
- [Dataset development](docs/datasets.html)
- [Model development](docs/models.html)
- [Training and evaluation](docs/training.html)
- [Experiment Manager](docs/experiment-manager.html)
- [Specialized utilities](docs/utilities.html)
- [API reference](docs/api.html)

## Project status

JimmyTorch is a personal research framework that evolves with active experiments. Review the current source and documentation together when adopting it in another project.

Licensed under the [Apache License 2.0](LICENSE).
