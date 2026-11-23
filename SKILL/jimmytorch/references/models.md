# Model contracts

Read this reference for `JimmyModel` subclasses, loss interfaces, reusable modules, mixed precision, or checkpoint changes.

## Required interface

A task model normally defines architecture plus:

- `train_loss_names`, including `Train/Main`.
- `eval_loss_names`, including `Eval/Main`.
- `forward(...)`.
- `trainStep(batch)` returning `(loss_dict, output_dict)` and calling `backwardOptimize(loss)`.
- `evalStep(batch)` returning the declared evaluation losses and detached outputs.
- `testStep(batch)` only when test behavior should differ; the base method performs a no-grad forward using `data` and `target` keys.

The trainer indexes every declared loss key. Do not declare metrics that a phase does not return. `evalStep` may include a matplotlib `Figure` under `output_dict["fig"]`; the trainer persists and closes it after evaluation.

## Initialization and optimization

Construct with `optimizer_cls`, `optimizer_args`, `mixed_precision`, `compile_model`, and `clip_grad`. Move the model to its device, optionally load a checkpoint, then call `initialize()` to compile and create the optimizer.

Use `getAutoCast(data_sample, self.mixed_precision)` around mixed-precision computation. `backwardOptimize` owns scaling, clipping, stepping, and gradient clearing; do not duplicate those operations around it.

`saveTo` writes a state dictionary. `loadFrom` keeps shape-compatible tensors and reports missing, unexpected, and mismatched keys. Treat any partial load as evidence to review, not silent success.

## Available components

- `Basics.py`: generated 1D/2D Conv-Norm-Activation and Norm-Activation-Conv blocks; `FCLayers`/`MLP`; sinusoidal, learned, and rotary positional encoders; patch makers.
- `Attentions.py`: `SELayer1D`, `SELayer2D`, `MHSA`, and cross-attention `MHCA`.
- `ModelUtils.py`: `Transpose`, `Permute`, `Reshape`, `PrintShape`, `SequentialMultiIO`, implemented `Rearrange`, `ResWrapper`, and `makeItResidual`.
- `LossFunctions.py`: `MaskedLoss`, `SequentialLossWithLength`, `RMSE`, and `KLDLoss`.
- `Functional.py`: `extendAs` and `getAutoCast` plus internal cost-analysis replacements.
- `CostTest.py`: `testCudaSpeed` and optional-`calflops` `testFlops`.

Inspect implementations before depending on tensor layout. In particular, generated convolution names encode dimension, normalization, activation, and ordering; `PosEncoderRotary` requires even feature dimension.

## Validation

- Run forward and each phase step on a tiny dictionary matching the dataset contract.
- Check all declared losses are finite Python-compatible scalar values.
- Confirm outputs used for logging are detached.
- Exercise mixed precision only on supported devices; keep a CPU-safe path.
- Round-trip a state dictionary and explicitly test any intended partial-load behavior.

## Opt-in distributed models

Use `DistributedJimmyModel` for distributed projects only; leave legacy `JimmyModel` projects unchanged. Preserve `trainStep` / `backwardOptimize`, using `self(...)` and one `self.backwardOptimize(loss)` per batch. All workers must execute matching control flow. The DDP bridge returns before backward, and saved weights have ordinary parameter names. No `computeTrainLoss` migration is needed. See `docs/distributed.html` for the supported contract and tests.
