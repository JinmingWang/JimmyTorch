# JimmyTorch API Reference

Complete API documentation for all JimmyTorch components.

---

## Dataset API

### JimmyDataset

| Method | Description |
|--------|-------------|
| `__init__(set_name, batch_size, drop_last, shuffle)` | Initialize dataset with batching parameters |
| `__getitem__(idx)` | Return batch dictionary for index idx (1-indexed) |
| `n_batches` | Property: number of batches based on n_samples |
| `__iter__()` | Initialize iteration state for batch loading |
| `__next__()` | Return next batch, raises StopIteration at end |
| `__len__()` | Return total number of samples |

### DatasetUtils

| Component | Type | Description |
|-----------|------|-------------|
| `DEVICE` | Variable | Global device (cuda/cpu) |

### MultiThreadLoader

| Method | Description |
|--------|-------------|
| `__init__(dataset, num_workers)` | Multi-threaded data loading for CPU-heavy preprocessing |

### TrajectoryUtils

| Function | Signature | Description |
|----------|-----------|-------------|
| `computeDistance` | `(trajs: BatchTraj \| Traj)` | Sum pair-wise distances along trajectory |
| `cropPadTraj` | `(traj, target_len, pad_value)` | Crop or pad trajectory to target length |
| `flipTrajWestEast` | `(trajs)` | Flip horizontally (negate longitude) |
| `centerTraj` | `(trajs)` | Center trajectory around origin (0,0) |
| `zScoreTraj` | `(trajs)` | Standardize to zero mean, unit variance |
| `rotateTraj` | `(trajs, angles)` | Rotate by given angles (degrees) |
| `interpTraj` | `(trajs, num_points, mode)` | Interpolate to num_points |
| `geometricDistance` | `(pred_points, gt_points, reduction)` | Haversine distance (meters) |
| `computeJSD` | `(dataset1, dataset2, num_bins)` | Jensen-Shannon Divergence |
| `plotTraj` | `(ax, trajs, traj_lengths, color)` | Plot trajectories on matplotlib axis |

---

## Model API

### JimmyModel

| Method | Description |
|--------|-------------|
| `__init__(optimizer_cls, optimizer_args, mixed_precision, compile_model, clip_grad)` | Initialize with training config |
| `initialize()` | Create optimizer, optionally compile |
| `trainStep(data_dict)` | Forward, backward, optimize. Return (loss_dict, output_dict) |
| `evalStep(data_dict)` | Evaluation with visualization. Return (loss_dict, output_dict) |
| `testStep(data_dict)` | Testing without gradients. Return (loss_dict, output_dict) |
| `backwardOptimize(loss)` | Backward with mixed precision + grad clipping |
| `saveTo(path)` | Save state_dict to path |
| `loadFrom(path)` | Load state_dict (handles size mismatches) |
| `getCompCost(self, *args, exit_after_print: bool=True, **kwargs)` | Get computational cost |
| `lr` | Property: current learning rate |

### Basic Layers (Models/Basics.py)

| Class | Description | Signature |
|-------|-------------|-----------|
| `Conv1DBnReLU` | Conv1D → BatchNorm → ReLU | `(c_in, c_out, k, s, p, d, g)` |
| `Conv2DBnGELU` | Conv2D → BatchNorm → GELU | `(c_in, c_out, k, s, p, d, g)` |
| `BnReLUConv1D` | BatchNorm → ReLU → Conv1D (pre-activation) | `(c_in, c_out, k, s, p, d, g)` |
| `FCLayers` / `MLP` | Multi-layer perceptron | `(channel_list, act, final_act)` |
| `PosEncoderSinusoidal` | Sinusoidal positional encoding | `(dim, max_len, merge_mode, d_pe)` |
| `PosEncoderLearned` | Learned positional encoding | `(dim, max_len, merge_mode)` |
| `PosEncoderRotary` | Rotary positional encoding (RoPE) | `(dim, max_len, base)` |
| `PatchMaker1D` | Extract 1D patches from sequences | `(patch_size, stride, patch_as_vector)` |
| `PatchMaker2D` | Extract 2D patches from images | `(patch_size, stride, patch_as_vector, flatten)` |

### Attention Modules (Models/Attentions.py)

| Class | Description | Signature |
|-------|-------------|-----------|
| `MHSA` | Multi-head self-attention | `(d_in, num_heads, dropout)` |
| `CrossAttention` | Cross-attention (Q vs KV) | `(d_in, num_heads, dropout)` |
| `SELayer1D` | Squeeze-and-Excitation 1D | `(c_in, reduction)` |
| `SELayer2D` | Squeeze-and-Excitation 2D | `(c_in, reduction)` |

### Model Utilities (Models/ModelUtils.py)

| Class | Description | Signature |
|-------|-------------|-----------|
| `Transpose` | Transpose dimensions | `(dim1, dim2)` |
| `Permute` | Permute dimensions | `(*dims)` |
| `Reshape` | Reshape tensor | `(*shape)` |
| `PrintShape` | Debug: print tensor shape | `(name)` |
| `SequentialMultiIO` | Sequential with multiple I/O | `forward(*dynamic, **static)` |

### Loss Functions (Models/LossFunctions.py)

| Class | Description | Signature |
|-------|-------------|-----------|
| `MaskedLoss` | Apply loss to masked elements | `(base_loss)` |
| `SequentialLossWithLength` | Loss for variable-length sequences | `(base_loss)` |
| `RMSE` | Root Mean Square Error | `()` |

### Functional (Models/Functional.py)

| Function | Description | Signature |
|----------|-------------|-----------|
| `getAutoCast` | Return autocast context for mixed precision | `(data, mixed_precision)` |

---

## Training API

### JimmyExperiment

| Method | Description |
|--------|-------------|
| `__init__(comments, dir_name)` | Initialize experiment with description |
| `start(checkpoint)` | Build components, launch training. Returns trainer |
| `test(model, test_set)` | Test model, return DataFrame report |

### JimmyTrainer

| Method | Description |
|--------|-------------|
| `__init__(train_set, eval_set, model, lr_scheduler, log_dir, save_dir, n_epochs, moving_avg, eval_interval, early_stop_lr)` | Initialize trainer |
| `start()` | Execute full training loop with logging |
| `evaluate(dataset, compute_avg)` | Evaluate model, return loss dict |

### DynamicConfig

| Method | Description |
|--------|-------------|
| `__init__(cls, **kwargs)` | Store class and initialization arguments |
| `build()` | Instantiate class with stored arguments |
| `add(**kwargs)` | Add or update arguments |
| `remove(key)` | Remove argument |

### ProgressManager (Training/ProgressManager.py)

| Method | Description |
|--------|-------------|
| `__init__(items_per_epoch, epochs, show_recent, refresh_interval, custom_fields)` | Initialize progress display |
| `update(epoch, step, **kwargs)` | Update display with current metrics |
| `close()` | Finalize progress display |

### MovingAverage (Training/MovingAverage.py)

| Method | Description |
|--------|-------------|
| `__init__(window_size)` | Initialize moving average tracker |
| `update(value)` | Add new value |
| `get()` | Retrieve current moving average |

### TensorBoardManager (Training/TensorBoardManager.py)

| Method | Description |
|--------|-------------|
| `__init__(log_dir, tags, value_types)` | Initialize TensorBoard logger |
| `log(step, **kwargs)` | Log scalar/histogram/image values |

---

## Utility Functions

### Diffusion Models

#### DDPM (DiffusionModels/DDPM.py)

| Method | Description |
|--------|-------------|
| `__init__(beta_min, beta_max, T, scale_mode, device)` | Initialize DDPM |
| `diffuse(x0, t, noise)` | Add noise to x0 at timestep t |
| `denoise(x_T, pred_func)` | Reverse diffusion from x_T to x_0 |

#### DDIM (DiffusionModels/DDIM.py)

| Method | Description |
|--------|-------------|
| `__init__(beta_min, beta_max, T, scale_mode, skip_step, device)` | Initialize DDIM |
| `diffuse(x0, t, noise)` | Add noise (same as DDPM) |
| `denoise(x_T, pred_func)` | Accelerated reverse diffusion |

---

## Type Definitions

### Trajectory Types

```python
Traj = FT32[Tensor, "L 2"]           # Single trajectory: (length, 2)
BatchTraj = FT32[Tensor, "B L 2"]    # Batch: (batch, length, 2)
```

### Loss Dict Format

```python
loss_dict = {
    "Train/Main": float,   # Required: main loss
    "Train/...": float,    # Optional: other training losses
    "Eval/Main": float,    # Required: main eval loss
    "Eval/...": float,     # Optional: other eval metrics
}
```

### Output Dict Format

```python
output_dict = {
    "output": Tensor,      # Model outputs (detached)
    "fig": Figure,         # Optional: matplotlib figure (evalStep only)
    ...                    # Other outputs
}
```

---

## Constants and Conventions

### Set Names

| Name | Purpose |
|------|---------|
| `train` | Training data (typically 70-80%) |
| `eval` | Validation data for LR scheduling |
| `test` | Final evaluation data |
| `debug` | Small subset for debugging |
| `all` | Complete dataset |

### Loss Name Requirements

- Must include a loss named `"Train/Main"` or `"Eval/Main"`
- Main loss controls LR scheduling and checkpointing
- Other losses are for logging only

### Positional Encoding Modes

| Mode | Description |
|------|-------------|
| `"add"` | Add encoding to input (same dim) |
| `"concat"` | Concatenate to input (increases dim) |

