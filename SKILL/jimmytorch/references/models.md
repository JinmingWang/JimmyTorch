# Model Implementation Guide

Source code in `JimmyTorch/Models/`

## Complete Implementation Pattern

```python
from JimmyTorch.Models import *
# Or if your components.py is defined and imports JimmyModel, import it directly:
# from .components import *

class MyModel(JimmyModel):  # Must inherit from JimmyModel
    def __init__(self, d_model: int, n_layers: int, **jimmy_model_kwargs):
        super().__init__(**jimmy_model_kwargs)

        # Define loss names
        # These 2 variable names are fixed
        # All losses names are in format "Phase/LossName"
        # Must have "Main" loss
        self.train_loss_names = ["Train/Main", "Train/MAE", "Train/MSE"]
        self.eval_loss_names = ["Eval/Main", "Eval/MAE", "Eval/MSE", "Eval/Acc", "Eval/JSD"]

        # Define architecture
        self.encoder = nn.Sequential(
            MLP([d_input, d_model, d_model], act=nn.ReLU()),
            PosEncoderSinusoidal(d_model, max_len=128, merge_mode='add'),
        )
        self.decoder = nn.Sequential(...)

        # Loss functions and metrics
        self.mse_func = nn.MSELoss()
        self.mae_func = nn.L1Loss()

    def forward(self, data):
        return self.decoder(self.encoder(data))

    def trainStep(self, batch: dict) -> tuple[dict, dict]:
        # batch is from JimmyDataset.__getitem__
        with getAutoCast(batch['data'], self.mixed_precision):
            output = self(batch['data'])
            mae = self.mae_func(output, batch['target'])
            mse = self.mse_func(output, batch['target'])
            loss = mae + mse

        self.backwardOptimize(loss)

        # Must return each loss, output must exist but can be empty
        return {
            "Train/Main": loss.item(),
            "Train/MAE": mae.item(),
            "Train/MSE": mse.item(),
        }, {
            "output": output.detach(),
        }

    def testStep(self, batch: dict) -> tuple[dict, dict]:
        with torch.no_grad():
            output = self(batch['data'])
            loss = self.mse_func(output, batch['target']).item()
            mae = self.mae_func(output, batch['target']).item()

        return {
            "Eval/Main": loss,
            "Eval/MAE": mae,
        }, {
            "output": output.detach(),
        }

    def evalStep(self, batch: dict) -> tuple[dict, dict]:
        losses, outputs = self.testStep(batch)
        
        # Add visualization
        fig, ax = plt.subplots()
        # ... plot outputs vs targets
        outputs["fig"] = fig
        
        return losses, outputs
```

## JimmyModel API

| Method | Description |
|--------|-------------|
| `__init__(optimizer_cls, optimizer_args, mixed_precision, compile_model, clip_grad)` | Initialize with training config |
| `initialize()` | Create optimizer, optionally compile |
| `trainStep(data_dict)` | Forward, backward, optimize. Return (loss_dict, output_dict) |
| `testStep(data_dict)` | Evaluate without gradients. Return (loss_dict, output_dict) |
| `evalStep(data_dict)` | Call testStep + add visualizations |
| `backwardOptimize(loss)` | Backward with mixed precision + grad clipping |
| `saveTo(path)` | Save state_dict |
| `loadFrom(path)` | Load state_dict (handles size mismatches) |
| `getCompCost(self, *args, exit_after_print: bool=True, **kwargs)` | Get computational cost |
| `lr` | Property: current learning rate |

## Building Blocks

### Basic Layers (Models/Basics.py)

| Class | Description | Signature |
|-------|-------------|-----------|
| `Conv1DBnReLU` | Conv1D → BN → ReLU | `(c_in, c_out, k, s, p, d, g)` |
| `Conv2DBnGELU` | Conv2D → BN → GELU | `(c_in, c_out, k, s, p, d, g)` |
| `BnReLUConv1D` | BN → ReLU → Conv1D (pre-act) | `(c_in, c_out, k, s, p, d, g)` |
| `MLP` / `FCLayers` | Multi-layer perceptron | `(channel_list, act, final_act)` |

**Naming Convention**: `[Norm][Act]Conv[ND]` (pre-act) or `Conv[ND][Norm][Act]` (post-act)

Example:
```python
encoder = nn.Sequential(
    Conv2DBnGELU(3, 64, 3, 1, 1),  # RGB → 64 channels
    Conv2DBnGELU(64, 128, 3, 2, 1),  # Downsample
    MLP([128, 256, 512], act=nn.GELU()),
)
```

### Positional Encodings

| Class | Mode | Description |
|-------|------|-------------|
| `PosEncoderSinusoidal` | add/concat | Fixed sinusoidal encoding |
| `PosEncoderLearned` | add/concat | Learned embedding table |
| `PosEncoderRotary` | rotary | RoPE (rotary positional encoding) |

Example:
```python
# Add positional encoding
encoder = nn.Sequential(
    nn.Linear(d_in, d_model),
    PosEncoderSinusoidal(d_model, max_len=128, merge_mode='add'),
)
```

### Attention Modules (Models/Attentions.py)

| Class | Description | Use Case |
|-------|-------------|----------|
| `MHSA` | Multi-head self-attention | Transformer encoder |
| `CrossAttention` | Cross-attention (Q vs KV) | Transformer decoder, conditioning |
| `SELayer1D` | Squeeze-and-Excitation 1D | Channel attention for sequences |
| `SELayer2D` | Squeeze-and-Excitation 2D | Channel attention for images |

Example:
```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attn = MHSA(d_model, num_heads, dropout=0.1)
        self.ffn = MLP([d_model, d_model * 4, d_model])
    
    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.ffn(x)
        return x
```

### Utility Layers (Models/ModelUtils.py)

For use in `nn.Sequential`:

| Class | Description |
|-------|-------------|
| `Transpose(dim1, dim2)` | Swap dimensions |
| `Permute(*dims)` | Reorder dimensions |
| `Reshape(*shape)` | Reshape tensor |
| `PrintShape(name)` | Debug: print shape |

Example:
```python
model = nn.Sequential(
    nn.Linear(128, 256),
    Transpose(1, 2),  # (B, 128, 256) → (B, 256, 128)
    Reshape(-1, 16, 16),  # Unflatten
)
```

### Loss Functions (Models/LossFunctions.py)

| Class | Description | Usage |
|-------|-------------|-------|
| `MaskedLoss(base_loss)` | Apply loss only to masked elements | Variable-length sequences |
| `SequentialLossWithLength(base_loss)` | Loss for variable-length sequences | Time-series with padding |
| `RMSE()` | Root mean square error | Regression metrics |

Example:
```python
# Masked loss for variable-length trajectories
self.mse_func = MaskedLoss(nn.MSELoss())

# In forward:
mask = (lengths > torch.arange(max_len)).float()
loss = self.mse_func(pred, target, mask)
```

## Advanced Features

### Mixed Precision Training

```python
model = MyModel(
    mixed_precision=True,  # Enable AMP
    clip_grad=1.0,  # Gradient clipping
)
```

Benefits: 2x faster, 50% less memory. Handles gradient scaling automatically.

### Model Compilation (PyTorch 2.0+)

```python
model = MyModel(compile_model=True)
```

Uses `torch.compile()` for optimization. Can speed up training ~20-30%.

### Gradient Clipping

```python
model = MyModel(clip_grad=1.0)  # Clip grad norm to 1.0
```

Applied after unscaling in mixed precision. Prevents exploding gradients.

### Checkpoint Loading with Size Mismatches

```python
model.loadFrom("checkpoint.pth")
# Automatically skips incompatible parameters
# Reports mismatches: "Skipped encoder.0.weight: size mismatch"
```

## Diffusion Models

### DDPM (Denoising Diffusion Probabilistic Models)

```python
from JimmyTorch.DiffusionModels import DDPM

ddm = DDPM(
    beta_min=0.0001,
    beta_max=0.02,
    T=1000,
    scale_mode="linear",  # or "quadratic"
    device=DEVICE
)

# Training
noisy = ddm.diffuse(x0, t, noise)
pred = model(noisy, t)

# Sampling
x_T = torch.randn_like(x0)
x_0 = ddm.denoise(x_T, pred_func)
```

### DDIM (Denoising Diffusion Implicit Models)

```python
from JimmyTorch.DiffusionModels import DDIM

ddm = DDIM(
    beta_min=0.0001,
    beta_max=0.02,
    T=1000,
    skip_step=10,  # Accelerated sampling
    device=DEVICE
)

# Same API as DDPM, but faster sampling
```

## Notes

- **Loss Names**: Must define `train_loss_names` and `eval_loss_names` with "Main" loss
- **Return Format**: `(loss_dict, output_dict)` from trainStep/testStep/evalStep
- **Mixed Precision**: Use `getAutoCast` context manager in forward pass
- **evalStep**: Must call testStep and add visualizations (figs) to output_dict
