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
- `DPMSolverPP`: `diffuseStep`, `diffuse`, `computeVelocity`, `samplingTimesteps`, `denoiseStep`, `denoise`
- `VPSDE`: `driftDiffusion`, `marginalProb`, `diffuse`, `computeScoreTarget`, `priorSampling`, `denoiseStep`, `denoise`
- `VESDE`: `driftDiffusion`, `marginalProb`, `diffuse`, `computeScoreTarget`, `priorSampling`, `denoiseStep`, `denoise`
- `PredictorCorrectorSampler`: `denoiseStep`, `denoise`

The discrete denoisers expect the prediction callback to return `(x0_pred, epsilon_0_to_t_pred, v_pred)` with exactly one non-`None` item. The `epsilon_pred` step keyword remains a compatibility alias. Existing camelCase method names remain canonical. `clean_timestep == -1` identifies the clean endpoint; schedule index `0` is the first noisy state. DDIM accepts `eta` in `[0, 1]`, with deterministic sampling at `eta=0`. DPM-Solver++ supports solver orders 1 and 2, generated or explicit descending timestep schedules, and uses a first-order data prediction for the final `0 -> clean_timestep` transition.

Score-SDE callbacks use `score_func(noisy_samples, continuous_timesteps, **kwargs) -> Tensor`, where the timestep tensor has shape `(batch,)` and values in `[epsilon, 1]`. The scalar `epsilon` is the terminal time cutoff, not a noise tensor. `PredictorCorrectorSampler` uses Euler-Maruyama prediction and zero or more Langevin correction steps. VP/VE perturbation helpers use `epsilon_0_to_t` for accumulated perturbation noise. Methods that sample noise accept an optional keyword-only `generator`.

## Training and monitoring

- `MovingAvg`, `MovingAvgGroup`, optional `TensorBoardManager`
- `ExperimentLogger`: local per-run SQLite writer
- `ExperimentManagerClient`: asynchronous live reporter and LR request poller
- `build_app`, `run_server`
- `DEFAULT_SERVER_HOST`, `DEFAULT_SERVER_PORT`, database filename constants, and lifecycle status constants

Package exports are defined by `Datasets/__init__.py`, `Models/__init__.py`, `DiffusionModels/__init__.py`, and `Training/ExperimentManager/__init__.py`. Do not infer exports from filenames alone.
