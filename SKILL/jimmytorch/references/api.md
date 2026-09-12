# Public API lookup

Use this as a name index, then inspect the named source before changing behavior or depending on an exact tensor layout.

## Core

- `DynamicConfig`: `build`, `add`, `remove`
- `JimmyExperiment`: `start`, `test`
- `JimmyTrainer`: `start`, `evaluate`
- `JimmyDataset`: `n_batches`, iteration, batch `__getitem__`
- `JimmyModel`: `initialize`, `backwardOptimize`, phase steps, `saveTo`, `loadFrom`, `getCompCost`, `lr`

## Models

- Basics: Conv/Norm/activation families, `FCLayers`/`MLP`, `PosEncoderSinusoidal`, `PosEncoderLearned`, `PosEncoderRotary`, `PatchMaker1D`, `PatchMaker2D`
- Attention: `SELayer1D`, `SELayer2D`, `MHSA`, `MHCA`
- Shape/composition: `Transpose`, `Permute`, `Reshape`, `PrintShape`, `SequentialMultiIO`, `Rearrange`, `ResWrapper`, `makeItResidual`
- Losses: `MaskedLoss`, `SequentialLossWithLength`, `RMSE`, `KLDLoss`
- Functional/cost: `extendAs`, `getAutoCast`, `testCudaSpeed`, `testFlops`

## Data

- Loading: `MultiThreadLoader`, `DEVICE`
- Sequence: `cropPadSequence`
- Trajectory: `getLng`, `getLat`, `computeDistance`, `cropPadTraj`, `flipTrajWestEast`, `flipTrajNorthSouth`, `centerTraj`, `zScoreTraj`, `minMaxTraj`, `rotateTraj`, `interpTraj`, `plotTraj`, `geometricDistance`, `computeJSD`
- 3D: `loadSTL`, `loadOFF`, `loadOBJ`, `loadPLY`, `meshSlicing`, `meshSlicing_vectorized`, `normalizeMesh`, `denormalizePoints`, `sampleAxis`, `buildPlaneBasis`, `projectPointsTo2D`, `reconstructPoints3D`, `computeSDFImages`, `loadModel`, `build_figure`

## Diffusion

- `DDPM`: `diffuseStep`, `diffuse`, `computeVelocity`, `denoiseStep`, `denoise`, `combineNoise`
- `DDIM`: `diffuseStep`, `diffuse`, `denoiseStep`, `denoise`, `extractNoise`, `computeVelocity`, `extractVelocity`

Both denoisers expect the prediction callback to return `(x0_pred, epsilon_pred, v_pred)`.

## Training and monitoring

- `MovingAvg`, `MovingAvgGroup`, optional `TensorBoardManager`
- `ExperimentLogger`: local per-run SQLite writer
- `ExperimentManagerClient`: asynchronous live reporter and LR request poller
- `build_app`, `run_server`
- `DEFAULT_SERVER_HOST`, `DEFAULT_SERVER_PORT`, database filename constants, and lifecycle status constants

Package exports are defined by `Datasets/__init__.py`, `Models/__init__.py`, `DiffusionModels/__init__.py`, and `Training/ExperimentManager/__init__.py`. Do not infer exports from filenames alone.
