# JimmyTorch

A personal PyTorch-based deep learning framework integrating dataset management, model training, experiment orchestration, and visualization. Optimized for research workflows with trajectory/sequential data.

[![en](https://img.shields.io/badge/lang-en-blue.svg)](README.md)
[![cn](https://img.shields.io/badge/lang-cn-red.svg)](README.cn.md)

---

## 1. Dataset

### Quick Usage

```python
from Datasets import JimmyDataset

class MyDataset(JimmyDataset):
    def __init__(self, batch_size: int, shuffle: bool = False):
        super().__init__(batch_size, drop_last=False, shuffle=shuffle)
        
        # Load and preprocess all data to tensors
        self.data = torch.randn(1000, 128).to(DEVICE)  # Example: 1000 samples
        self.labels = torch.randint(0, 10, (1000,)).to(DEVICE)
        self.n_samples = len(self.data)  # Must set n_samples
    
    def __getitem__(self, idx):
        start = (idx - 1) * self.batch_size
        end = min(start + self.batch_size, self.n_samples)
        indices = self._indices[start:end]
        
        return {
            'data': self.data[indices],
            'target': self.labels[indices]
        }

# Usage
train_set = MyDataset(batch_size=32, shuffle=True)
for batch_dict in train_set:
    data = batch_dict['data']
    target = batch_dict['target']
```

### Table of Contents

| Path | Class | Function | Description |
|------|-------|----------|-------------|
| `Datasets/JimmyDataset.py` | `JimmyDataset` | - | Base dataset class combining Dataset + DataLoader functionality |
| `Datasets/JimmyDataset.py` | `JimmyDataset` | `__init__(batch_size, drop_last, shuffle)` | Initialize dataset with batching parameters |
| `Datasets/JimmyDataset.py` | `JimmyDataset` | `__getitem__(idx)` | Return batch dictionary for index idx (1-indexed) |
| `Datasets/JimmyDataset.py` | `JimmyDataset` | `n_batches` | Property computing number of batches based on n_samples |
| `Datasets/DatasetUtils.py` | - | `DEVICE` | Global device variable (cuda/cpu) |
| `Datasets/MultiThreadLoader.py` | `MultiThreadLoader` | `__init__(dataset, num_workers)` | Multi-threaded data loading for CPU-heavy preprocessing |
| `Datasets/TrajectoryUtils.py` | - | `computeDistance(trajs: BatchTraj \| Traj)` | Compute total distance of trajectory by summing pair-wise distances |
| `Datasets/TrajectoryUtils.py` | - | `cropPadTraj(traj, target_len, pad_value)` | Crop or pad trajectory to target length |
| `Datasets/TrajectoryUtils.py` | - | `flipTrajWestEast(trajs)` | Flip trajectory horizontally by negating longitude |
| `Datasets/TrajectoryUtils.py` | - | `centerTraj(trajs)` | Center trajectory around origin (0, 0) |
| `Datasets/TrajectoryUtils.py` | - | `zScoreTraj(trajs)` | Standardize trajectory to zero mean and unit variance |
| `Datasets/TrajectoryUtils.py` | - | `rotateTraj(trajs, angles)` | Rotate trajectory by given angles (degrees) |
| `Datasets/TrajectoryUtils.py` | - | `interpTraj(trajs, num_points, mode)` | Interpolate trajectory to num_points using specified mode |
| `Datasets/TrajectoryUtils.py` | - | `geometricDistance(pred_points, gt_points, reduction)` | Compute Haversine distance between GPS points in meters |
| `Datasets/TrajectoryUtils.py` | - | `computeJSD(dataset1, dataset2, num_bins)` | Compute Jensen-Shannon Divergence between trajectory distributions |
| `Datasets/TrajectoryUtils.py` | - | `plotTraj(ax, trajs, traj_lengths, color)` | Plot trajectories on matplotlib axis |
| `Datasets/SequenceUtils.py` | - | - | (TODO) Utility functions for sequential data |
| `Datasets/MNISTDataset.py` | `MNISTSampleDataset` | - | Example dataset implementation for MNIST |

### Notes

- **Preloading Philosophy**: Unlike PyTorch's DataLoader, `JimmyDataset` assumes all data is preprocessed into tensors and loaded to GPU during `__init__`. This eliminates multi-threaded loading overhead when data fits in memory.
- **Dictionary Return**: `__getitem__` must return a dictionary. This unifies interfaces across different datasets and models.
- **Index Convention**: `__getitem__(idx)` uses 1-indexed batches (idx=1 is first batch). Internal implementation handles 0-indexing conversion.
- **Trajectory Types**: `Traj = FT32[Tensor, "L 2"]` (single trajectory), `BatchTraj = FT32[Tensor, "B L 2"]` (batch of trajectories).
- **MultiThreadLoader**: Use only when preprocessing contains unavoidable CPU-heavy operations (e.g., file I/O, complex transformations).

---

## 2. Model

### Quick Usage

```python
from Models import JimmyModel
import torch.nn as nn

class MyModel(JimmyModel):
    def __init__(self):
        super().__init__(
            optimizer_cls=torch.optim.AdamW,
            optimizer_args={"lr": 1e-4, "weight_decay": 0.01},
            mixed_precision=True,
            compile_model=False,
            clip_grad=1.0
        )
        
        # Define loss names for logging
        self.train_loss_names = ["Train/Loss", "Train/Acc"]
        self.eval_loss_names = ["Eval/Loss", "Eval/Acc"]
        
        # Define model architecture
        self.encoder = nn.Sequential(...)
        self.decoder = nn.Sequential(...)
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, x):
        return self.decoder(self.encoder(x))
    
    def trainStep(self, data_dict):
        with getAutoCast(data_dict['data'], self.mixed_precision):
            output = self(data_dict['data'])
            loss = self.loss_fn(output, data_dict['target'])
            acc = (output.argmax(dim=-1) == data_dict['target']).float().mean()
        
        self.backwardOptimize(loss)
        
        return {"Train/Loss": loss.item(), "Train/Acc": acc.item()}, \
               {"output": output.detach()}
    
    def evalStep(self, data_dict):
        with torch.no_grad():
            output = self(data_dict['data'])
            loss = self.loss_fn(output, data_dict['target']).item()
            acc = (output.argmax(dim=-1) == data_dict['target']).float().mean().item()
        
        return {"Eval/Loss": loss, "Eval/Acc": acc}, \
               {"output": output.detach()}

# Usage
model = MyModel().to(DEVICE)
model.initialize()  # Create optimizer
```

### Table of Contents

| Path | Class | Function | Description |
|------|-------|----------|-------------|
| `Models/JimmyModel.py` | `JimmyModel` | `__init__(optimizer_cls, optimizer_args, mixed_precision, compile_model, clip_grad)` | Initialize model with training configuration |
| `Models/JimmyModel.py` | `JimmyModel` | `initialize()` | Create optimizer and optionally compile model |
| `Models/JimmyModel.py` | `JimmyModel` | `trainStep(data_dict)` | Execute one training step: forward, backward, optimize. Return loss_dict and output_dict |
| `Models/JimmyModel.py` | `JimmyModel` | `evalStep(data_dict)` | Execute one evaluation step. Return loss_dict and output_dict |
| `Models/JimmyModel.py` | `JimmyModel` | `testStep(data_dict)` | Execute one test step (default: calls evalStep) |
| `Models/JimmyModel.py` | `JimmyModel` | `backwardOptimize(loss)` | Perform backward pass with mixed precision and gradient clipping |
| `Models/JimmyModel.py` | `JimmyModel` | `saveTo(path)` | Save model state_dict to path |
| `Models/JimmyModel.py` | `JimmyModel` | `loadFrom(path)` | Load model state_dict from path with size mismatch handling |
| `Models/JimmyModel.py` | `JimmyModel` | `lr` | Property returning current learning rate |
| `Models/Basics.py` | `Conv1DBnReLU` | `__init__(c_in, c_out, k, s, p, d, g)` | 1D Conv → BatchNorm → ReLU block |
| `Models/Basics.py` | `Conv2DBnGELU` | `__init__(c_in, c_out, k, s, p, d, g)` | 2D Conv → BatchNorm → GELU block |
| `Models/Basics.py` | `BnReLUConv1D` | `__init__(c_in, c_out, k, s, p, d, g)` | BatchNorm → ReLU → 1D Conv block (pre-activation) |
| `Models/Basics.py` | `FCLayers` / `MLP` | `__init__(channel_list, act, final_act)` | Multi-layer perceptron with configurable activation |
| `Models/Basics.py` | `PosEncoderSinusoidal` | `__init__(dim, max_len, merge_mode, d_pe)` | Sinusoidal positional encoding (add/concat) |
| `Models/Basics.py` | `PosEncoderLearned` | `__init__(dim, max_len, merge_mode)` | Learned positional encoding |
| `Models/Basics.py` | `PosEncoderRotary` | `__init__(dim, max_len, base)` | Rotary positional encoding (RoPE) |
| `Models/Basics.py` | `PatchMaker1D` | `__init__(patch_size, stride, patch_as_vector)` | Extract 1D patches from sequences |
| `Models/Basics.py` | `PatchMaker2D` | `__init__(patch_size, stride, patch_as_vector, flatten)` | Extract 2D patches from images |
| `Models/Attentions.py` | `MHSA` | `__init__(d_in, num_heads, dropout)` | Multi-head self-attention with QKV projections |
| `Models/Attentions.py` | `CrossAttention` | `__init__(d_in, num_heads, dropout)` | Cross-attention between query and key-value pairs |
| `Models/Attentions.py` | `SELayer1D` | `__init__(c_in, reduction)` | Squeeze-and-Excitation for 1D features |
| `Models/Attentions.py` | `SELayer2D` | `__init__(c_in, reduction)` | Squeeze-and-Excitation for 2D features |
| `Models/ModelUtils.py` | `Transpose` | `__init__(dim1, dim2)` | Transpose layer for nn.Sequential |
| `Models/ModelUtils.py` | `Permute` | `__init__(*dims)` | Permute layer for nn.Sequential |
| `Models/ModelUtils.py` | `Reshape` | `__init__(*shape)` | Reshape layer for nn.Sequential |
| `Models/ModelUtils.py` | `PrintShape` | `__init__(name)` | Debug layer to print tensor shape |
| `Models/ModelUtils.py` | `SequentialMultiIO` | `forward(*dynamic_inputs, **static_inputs)` | Sequential module supporting multiple inputs/outputs |
| `Models/ModelUtils.py` | `Rearrange` | - | Einops-like rearrangement (implementation pending) |
| `Models/LossFunctions.py` | `MaskedLoss` | `__init__(base_loss)` | Apply loss only to masked elements |
| `Models/LossFunctions.py` | `SequentialLossWithLength` | `__init__(base_loss)` | Apply loss to sequences with variable lengths |
| `Models/LossFunctions.py` | `RMSE` | - | Root Mean Square Error loss |
| `Models/Functional.py` | - | `getAutoCast(data, mixed_precision)` | Return autocast context for mixed precision |
| `DiffusionModels/DDPM.py` | `DDPM` | - | Denoising Diffusion Probabilistic Models implementation |
| `DiffusionModels/DDIM.py` | `DDIM` | - | Denoising Diffusion Implicit Models implementation |

### Notes

- **Loss Names**: `train_loss_names` and `eval_loss_names` must be defined. They control what metrics are logged and visualized.
- **Return Format**: `trainStep` and `evalStep` must return `(loss_dict, output_dict)`. Keys in `loss_dict` must match the declared loss names.
- **Mixed Precision**: Handles gradient scaling automatically. Use `getAutoCast` context in forward pass.
- **Gradient Clipping**: Set `clip_grad > 0` to enable. Applied after unscaling gradients in mixed precision.
- **Model Compilation**: `compile_model=True` uses `torch.compile()` for optimization (PyTorch 2.0+).
- **Checkpoint Loading**: `loadFrom` gracefully handles size mismatches by skipping incompatible parameters and reporting them.
- **Basics Module Naming Convention**: Format is `[Norm][Act]Conv[ND]` (pre-activation) or `Conv[ND][Norm][Act]` (post-activation). Example: `Conv2DBnGELU` = Conv2D → BatchNorm → GELU.
- **Positional Encoding Merge Modes**: `"add"` adds encoding to input; `"concat"` concatenates along feature dimension.

---

## 3. Training and Experiment

### Quick Usage

```python
from JimmyExperiment import JimmyExperiment
from DynamicConfig import DynamicConfig

# Define experiment configuration
exp = JimmyExperiment(comments="Baseline experiment with AdamW")

# Configure dataset
exp.dataset_cfg = DynamicConfig(
    MyDataset,
    batch_size=64,
    shuffle=True
)

# Configure model
exp.model_cfg = DynamicConfig(
    MyModel,
    optimizer_cls=torch.optim.AdamW,
    optimizer_args={"lr": 1e-4},
    mixed_precision=True
)

# Configure learning rate scheduler
exp.lr_scheduler_cfg = DynamicConfig(
    JimmyLRScheduler,
    peak_lr=2e-4,
    min_lr=1e-7,
    warmup_count=10,
    patience=10,
    decay_rate=0.5
)

# Set training constants
exp.constants = {
    "n_epochs": 100,
    "moving_avg": 100,
    "eval_interval": 5,
    "early_stop_lr": 1e-6,  # Optional: early stopping threshold
    "save_dir": None,  # Optional: custom checkpoint directory
    "log_dir": None  # Optional: custom log directory
}

# Start training
trainer = exp.start(checkpoint=None)

# Optional: Test separately after training
exp.dataset_cfg.set_name = "test"
test_set = exp.dataset_cfg.build()
test_report = exp.test(trainer.model, test_set)
test_report.to_csv("test_results.csv")
```

### Table of Contents

| Path | Class | Function | Description |
|------|-------|----------|-------------|
| `JimmyTrainer.py` | `JimmyTrainer` | `__init__(train_set, eval_set, model, lr_scheduler, log_dir, save_dir, n_epochs, moving_avg, eval_interval, early_stop_lr)` | Initialize trainer with datasets, model, and hyperparameters |
| `JimmyTrainer.py` | `JimmyTrainer` | `start()` | Execute full training loop with logging and checkpointing |
| `JimmyTrainer.py` | `JimmyTrainer` | `evaluate(dataset, compute_avg)` | Evaluate model on dataset, return loss dict (averaged or per-sample) |
| `JimmyExperiment.py` | `JimmyExperiment` | `__init__(comments, dir_name)` | Initialize experiment with description and optional custom directory name |
| `JimmyExperiment.py` | `JimmyExperiment` | `start(checkpoint)` | Build components from configs and launch training, returns trainer |
| `JimmyExperiment.py` | `JimmyExperiment` | `test(model, test_set)` | Test model on dataset and return detailed report DataFrame |
| `DynamicConfig.py` | `DynamicConfig` | `__init__(cls, **kwargs)` | Store class and initialization arguments |
| `DynamicConfig.py` | `DynamicConfig` | `build()` | Instantiate class with stored arguments |
| `DynamicConfig.py` | `DynamicConfig` | `add(**kwargs)` | Add or update arguments |
| `DynamicConfig.py` | `DynamicConfig` | `remove(key)` | Remove argument |
| `Training/JimmyLRScheduler.py` | `JimmyLRScheduler` | `__init__(optimizer, peak_lr, min_lr, warmup_count, window_size, patience, decay_rate)` | Initialize adaptive learning rate scheduler |
| `Training/JimmyLRScheduler.py` | `JimmyLRScheduler` | `update(metric)` | Update learning rate based on metric (loss) |
| `Training/JimmyLRScheduler.py` | `JimmyLRScheduler` | `phaseWarmUP()` | Compute LR during warmup phase (sinusoidal ramp) |
| `Training/JimmyLRScheduler.py` | `JimmyLRScheduler` | `phaseCosine()` | Apply cosine annealing to LR |
| `Training/JimmyLRScheduler.py` | `JimmyLRScheduler` | `phaseDecay()` | Decay LR when metric plateaus |
| `Training/ProgressManager.py` | `ProgressManager` | `__init__(items_per_epoch, epochs, show_recent, refresh_interval, custom_fields)` | Initialize progress visualization with custom metric fields |
| `Training/ProgressManager.py` | `ProgressManager` | `update(epoch, step, **kwargs)` | Update progress display with current metrics |
| `Training/ProgressManager.py` | `ProgressManager` | `close()` | Finalize and close progress display |
| `Training/MovingAverage.py` | `MovingAvg` | `__init__(window_size)` | Initialize moving average tracker |
| `Training/MovingAverage.py` | `MovingAvg` | `update(value)` | Add new value to moving average |
| `Training/MovingAverage.py` | `MovingAvg` | `get()` | Retrieve current moving average |
| `Training/TensorBoardManager.py` | `TensorBoardManager` | `__init__(log_dir, tags, value_types)` | Initialize TensorBoard logger with pre-registered tags |
| `Training/TensorBoardManager.py` | `TensorBoardManager` | `log(step, **kwargs)` | Log scalar/histogram/image values to TensorBoard |
| `RunRecordManager.py` | `RunRecordManager` | - | Manage experiment records and results (persistence layer) |

### Notes

- **DynamicConfig Pattern**: Defers object instantiation until `build()` is called. Allows runtime modification of hyperparameters without recreating objects.
- **Custom Directory Naming**: Pass `dir_name` parameter to `JimmyExperiment.__init__()` for meaningful run names instead of timestamps. If not provided, uses timestamp format `%y%m%d_%H%M%S`.
- **Separated Testing**: Training and testing are now decoupled. `start()` only trains; use `test()` method for flexible evaluation on multiple configurations.
- **Runtime Parameter Hot-Reload**: Training creates `runtime_param_buffer.yaml` in log directory. Edit this file during training to adjust learning rate on-the-fly (changes apply on next epoch).
- **Trainer Type Selection**: Set `experiment.trainer_type` to use custom trainer classes for different training paradigms (e.g., iteration-based, GAN training).
- **Early Stopping**: Set `early_stop_lr` in constants to automatically stop training when learning rate drops below threshold.
- **JimmyLRScheduler Phases**: (1) Warmup with sinusoidal ramp to peak_lr, (2) Cosine annealing with high-frequency modulation, (3) Exponential decay on plateau detection.
- **Plateau Detection**: LR scheduler tracks moving average of metric over `window_size` epochs. If no improvement for `patience` epochs, triggers decay.
- **Automatic Directory Structure**: Experiment creates `Runs/{DatasetName}/{ModelName}/{dir_name}/` for logs, checkpoints, and configs. Custom paths via `save_dir` and `log_dir` in constants.
- **Checkpoint Strategy**: Saves `best.pth` (lowest eval loss) and `last.pth` (most recent) every `eval_interval` epochs.
- **ProgressManager**: Uses Rich library for live-updating terminal display. Shows recent N epochs, ETA, and custom metrics.
- **Metric Logging**: Training metrics use moving average (reduces noise), evaluation metrics are raw values.
- **PyTorch LR Scheduler Compatibility**: `JimmyTrainer` auto-wraps PyTorch schedulers by detecting `step()` signature and creating compatible `update()` method.

---

## 4. Overall Pipeline

1. **Implement Dataset**: Inherit from `JimmyDataset`, load all data to tensors in `__init__`, set `self.n_samples`, implement `__getitem__` returning dictionary.

2. **Implement Model**: Inherit from `JimmyModel`, define architecture and `train_loss_names`/`eval_loss_names` in `__init__`, implement `forward`, `trainStep`, `evalStep`.

3. **Configure Experiment**: Create `JimmyExperiment` instance, configure `dataset_cfg`, `model_cfg`, `lr_scheduler_cfg` using `DynamicConfig`, set training constants (`n_epochs`, `eval_interval`, etc.).

4. **Launch Training**: Call `experiment.start(checkpoint=None)` to build components, create directories, and execute training loop. Returns trainer object for further use.

5. **Monitor Progress**: View real-time progress in terminal via `ProgressManager`, check TensorBoard logs at `Runs/{DatasetName}/{ModelName}/{dir_name}/`. During training, edit `runtime_param_buffer.yaml` to adjust learning rate.

6. **Evaluate and Test**: After training, call `experiment.test(model, test_set)` to evaluate on test set and get detailed DataFrame report. Save with `test_report.to_csv()`. Can test multiple configurations without retraining.

7. **Customize for New Tasks**: For different training pipelines (e.g., GANs, reinforcement learning), implement custom `Trainer` and `Experiment` classes following the same patterns. Use `main.py` as template for entry scripts.

8. **Utilize Building Blocks**: Compose models using components from `Models/Basics.py` (Conv blocks, MLPs, positional encodings), `Models/Attentions.py` (MHSA, cross-attention, SE layers), and `Models/ModelUtils.py` (shape manipulation layers).

9. **Handle Trajectories**: For GPS trajectory tasks, use utilities from `Datasets/TrajectoryUtils.py` for preprocessing (interpolation, rotation, normalization) and evaluation (geometric distance, JSD).

10. **Leverage Advanced Features**: Enable mixed precision training (`mixed_precision=True`), gradient clipping (`clip_grad > 0`), model compilation (`compile_model=True`), and adaptive LR scheduling (`JimmyLRScheduler`) for efficient training.

---

## Key Design Principles

- **Dictionary Interfaces**: Datasets return dicts, models accept/return dicts. Enables flexible composition without rigid argument ordering.
- **Separation of Concerns**: Dataset handles data loading, Model handles computation and optimization, Trainer orchestrates loops, Experiment manages configurations.
- **GPU-First Philosophy**: Assumes data fits in GPU memory. Eliminates CPU-GPU transfer overhead during training.
- **Configuration as Code**: Use `DynamicConfig` to version and modify hyperparameters programmatically.
- **Reproducibility**: Experiment logs save model architecture, hyperparameters, and comments automatically.
- **Extensibility**: Core classes (`JimmyDataset`, `JimmyModel`, `JimmyTrainer`, `JimmyExperiment`) are templates. Override methods for task-specific logic.

---

## Quick Start Example

```python
# 1. Dataset
from Datasets import JimmyDataset
class MyData(JimmyDataset):
    def __init__(self, batch_size):
        super().__init__(batch_size)
        self.X = torch.randn(1000, 10).to(DEVICE)
        self.y = torch.randint(0, 2, (1000,)).to(DEVICE)
        self.n_samples = 1000
    def __getitem__(self, idx):
        start, end = (idx-1)*self.batch_size, idx*self.batch_size
        return {'data': self.X[start:end], 'target': self.y[start:end]}

# 2. Model
from Models import JimmyModel
import torch.nn as nn
class MyModel(JimmyModel):
    def __init__(self):
        super().__init__(optimizer_cls=torch.optim.Adam, optimizer_args={"lr": 1e-3})
        self.train_loss_names = ["Train/Loss"]
        self.eval_loss_names = ["Eval/Loss"]
        self.net = nn.Linear(10, 2)
        self.loss_fn = nn.CrossEntropyLoss()
    def forward(self, x): return self.net(x)
    def trainStep(self, d):
        out = self(d['data'])
        loss = self.loss_fn(out, d['target'])
        self.backwardOptimize(loss)
        return {"Train/Loss": loss.item()}, {"output": out.detach()}
    def evalStep(self, d):
        with torch.no_grad():
            out = self(d['data'])
            loss = self.loss_fn(out, d['target'])
        return {"Eval/Loss": loss.item()}, {"output": out}

# 3. Experiment
from JimmyExperiment import JimmyExperiment
from DynamicConfig import DynamicConfig
exp = JimmyExperiment("Quick start test")
exp.dataset_cfg = DynamicConfig(MyData, batch_size=32)
exp.model_cfg = DynamicConfig(MyModel)
exp.constants = {"n_epochs": 10, "moving_avg": 10, "eval_interval": 2}
trainer = exp.start()
```

This example demonstrates the complete pipeline in ~40 lines of code.
