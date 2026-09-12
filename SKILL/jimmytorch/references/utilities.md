# Specialized utility guidance

Read only the relevant section when changing diffusion, trajectory, sequence, model-cost, or 3D helpers.

## Diffusion

`DDPM` and `DDIM` use `min_beta`, `max_beta`, `max_diffusion_step`, `device`, and `scale_mode`; DDIM also has `skip_step`. Schedules support `linear`, `quadratic`, and `log`.

Both flatten non-batch dimensions internally and restore the original shape. Their `denoise` callbacks return `(x0_pred, epsilon_pred, v_pred)`. Preserve device placement, scalar-versus-batched timestep behavior, final-step handling, and optional `need_x0` return shape.

DDIM additionally exposes `extractNoise` and `extractVelocity`; DDPM exposes `combineNoise` and backward-compatible schedule aliases.

## Trajectories and sequences

Trajectory tensors are `(L, 2)` or `(B, L, 2)`. Check whether each function accepts one or both forms. `computeDistance` uses Euclidean coordinate units; `geometricDistance` uses longitude/latitude and returns metres. `cropPadSequence` crops/pads dimension zero for general tensors.

Test degenerate lengths, batched/unbatched parity, dtype/device preservation, interpolation endpoints, reduction modes, and numerical stability of normalization.

## 3D geometry

Loaders produce triangle tensors shaped `(F, 3, 3)`. OFF/OBJ polygon faces are fan-triangulated; PLY supports ASCII only; STL needs `numpy-stl`. The usual pipeline is load → normalize → slice → project → rasterize/view.

`computeSDFImages` expects `(N, S_max, 2, 2)` padded segment data and `(N,)` counts, chunks over slices and pixel rows, and returns CPU float32 output. Preserve nested-contour sign behavior and memory bounds.

## Model cost

`testCudaSpeed` requires CUDA. `testFlops` depends on optional `calflops` and replaces attention/encoder layers internally for analysis. Do not make these optional diagnostics a requirement for ordinary model imports or training.
