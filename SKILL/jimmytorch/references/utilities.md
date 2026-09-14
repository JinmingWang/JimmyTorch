# Specialized utility guidance

Read only the relevant section when changing diffusion, trajectory, sequence, model-cost, or 3D helpers.

## Diffusion

`DDPM`, `DDIM`, and `DPMSolverPP` use the internal discrete Gaussian hierarchy. They accept `min_beta`, `max_beta`, `max_diffusion_step`, `device`, and `scale_mode`; DDIM also has `skip_step` and `eta`, while DPM-Solver++ has `num_inference_steps` and `solver_order`. Schedules support `linear`, `quadratic`, `log`, and the Improved-DDPM `cosine` schedule. The cosine schedule derives beta values from its cumulative-alpha curve rather than `min_beta` and `max_beta`.

All three use rank-aware coefficient broadcasting without flattening samples. Their `denoise` callbacks return `(x0_pred, epsilon_0_to_t_pred, v_pred)` with exactly one prediction populated. Use `epsilon_t_to_tp1` only for single-transition noise, `epsilon_0_to_t` for accumulated noise at the current level, and `epsilon_0_to_tp1` for the combined accumulated result. The old `noise`, `epsilon`, `epsilon_pred`, and abbreviated `eps_*` keywords remain compatibility aliases where they were previously public.

Schedule index `0` is the first noisy state, while `clean_timestep == -1` represents clean data. DDPM reverse steps must be adjacent; DDIM and DPM-Solver++ can skip indices. DPM-Solver++ explicit inference schedules start at `T-1`, descend strictly, and end at `0`; order two keeps one previous data prediction as multistep history, while the final clean transition is a stable first-order data prediction. Preserve device and dtype placement, scalar-versus-batched timestep behavior, final-step handling, optional `need_x0` return shape, and seeded `generator` behavior.

Direct `denoiseStep` calls that pass `t_prev=0` emit `DeprecationWarning` because older releases treated zero as clean data. Use `clean_timestep` for the final transition. A process and all input, prediction, noise, and timestep tensors must be on the configured device.

DDIM additionally exposes `extractNoise` and `extractVelocity`; DDPM exposes `combineNoise` and backward-compatible schedule aliases. DDIM is deterministic when `eta=0`; positive `eta` values up to `1` add stochastic noise. DPM-Solver++ is deterministic and its stateless `denoiseStep` is first order; use `denoise` for second-order multistep history.

`VPSDE` and `VESDE` use a separate continuous-time hierarchy with times in `[0, 1]`. Their public methods are `driftDiffusion`, `marginalProb`, `diffuse`, `computeScoreTarget`, `priorSampling`, `denoiseStep`, and `denoise`. Perturbations use accumulated `epsilon_0_to_t`, and the analytic score target is `-epsilon_0_to_t / std(t)`. Do not mix these continuous times with discrete schedule indices.

`PredictorCorrectorSampler(sde, num_steps, corrector_steps, snr, epsilon)` integrates from `1` down to the positive terminal `epsilon`. Here `epsilon` is a scalar terminal time, not a noise tensor; perturbation noise keeps the range-explicit name `epsilon_0_to_t`. Its callback contract is `score_func(noisy_samples, continuous_timesteps, **kwargs) -> Tensor`; the timestep input is always batch-shaped. It uses Euler-Maruyama prediction, and `corrector_steps=0` disables Langevin correction. Noise sampling accepts a keyword-only `generator`.

Canonical references are the DDPM paper (https://arxiv.org/abs/2006.11239), DDIM paper (https://arxiv.org/abs/2010.02502), Improved-DDPM paper (https://arxiv.org/abs/2102.09672), Score-SDE paper (https://arxiv.org/abs/2011.13456), and DPM-Solver++ paper (https://arxiv.org/abs/2211.01095).

## Trajectories and sequences

Trajectory tensors are `(L, 2)` or `(B, L, 2)`. Check whether each function accepts one or both forms. `computeDistance` uses Euclidean coordinate units; `geometricDistance` uses longitude/latitude and returns metres. `cropPadSequence` crops/pads dimension zero for general tensors.

Test degenerate lengths, batched/unbatched parity, dtype/device preservation, interpolation endpoints, reduction modes, and numerical stability of normalization.

## 3D geometry

Loaders produce triangle tensors shaped `(F, 3, 3)`. OFF/OBJ polygon faces are fan-triangulated; PLY supports ASCII only; STL needs `numpy-stl`. The usual pipeline is load → normalize → slice → project → rasterize/view.

`computeSDFImages` expects `(N, S_max, 2, 2)` padded segment data and `(N,)` counts, chunks over slices and pixel rows, and returns CPU float32 output. Preserve nested-contour sign behavior and memory bounds.

## Model cost

`testCudaSpeed` requires CUDA. `testFlops` depends on optional `calflops` and replaces attention/encoder layers internally for analysis. Do not make these optional diagnostics a requirement for ordinary model imports or training.
