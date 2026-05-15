# Dataset Implementation Guide

Source code in `JimmyTorch/Datasets/`

## Complete Implementation Pattern

```python
from JimmyTorch.Datasets import *   # This line is fixed

class MyDataset(JimmyDataset):  # Must inherit from JimmyDataset
    def __init__(self,
                data_path: str,
                set_name: Literal['train', 'eval', 'test', 'debug', 'all'], 
                batch_size: int,
                drop_last: bool = False,
                shuffle: bool = False, 
                **dataset_specific_args):
        # Must call super().__init__()
        super().__init__(set_name, batch_size, drop_last, shuffle)

        # Slice data according to set_name
        slices = {
            'train': slice(0, -300),
            'eval': slice(-300, -200),
            'test': slice(-200, None),
            'debug': slice(0, 100),
            'all': slice(0, None)
        }
        slicing = slices[set_name]
        
        # Load data (expect dict)
        dataset = torch.load(data_path)
        
        # Unpack dataset and assign to attributes
        # Try to keep all data on GPU for zero-overhead access during training
        # Be careful about GPU memory size!!!
        self.traj = dataset['traj'].to(DEVICE)[slicing]
        self.map = dataset['map'].to(DEVICE)[slicing]
        
        # Must set n_samples
        self.n_samples = len(self.traj)
    
    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min(start + self.batch_size, self.n_samples)
        indices = self._indices[start:end]

        batch_traj = self.traj[indices]
        batch_map = self.map[indices]

        # Data augmentation if necessary
        # Batch-level preprocessing if necessary
        
        # Must return dict containing ALL available data types
        # Never use keys like "x", "y", "input", "target", too ambiguous.
        return {
            'traj': batch_traj,
            'map': batch_map,
        }
```

## JimmyDataset API

| Method | Description |
|--------|-------------|
| `__init__(batch_size, drop_last, shuffle)` | Initialize with batching parameters |
| `__getitem__(idx)` | Return batch dict for index idx |
| `n_batches` | Property: number of batches |
| `__iter__()` | Initialize iteration state |
| `__next__()` | Return next batch, raises StopIteration at end |
| `__len__()` | Total number of samples (*NOT* batches) |

## MultiThreadLoader

Use only when preprocessing contains unavoidable CPU-heavy operations:

```python
from JimmyTorch.Datasets import MultiThreadLoader

train_set = MyDataset('train', batch_size=64)
loader = MultiThreadLoader(train_set, num_workers=4)

for batch in loader:
    # batch is already on GPU
    pass
```

## Trajectory Utilities

### Type Definitions
- `Traj = FT32[Tensor, "L 2"]` - Single trajectory (length × coordinates)
- `BatchTraj = FT32[Tensor, "B L 2"]` - Batch of trajectories

### Available Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `computeDistance` | `(trajs: BatchTraj | Traj)` | Sum of pairwise distances |
| `cropPadTraj` | `(traj, target_len, pad_value)` | Crop or pad to length |
| `flipTrajWestEast` | `(trajs)` | Flip horizontally (negate lon) |
| `centerTraj` | `(trajs)` | Center around (0, 0) |
| `zScoreTraj` | `(trajs)` | Standardize to μ=0, σ=1 |
| `rotateTraj` | `(trajs, angles)` | Rotate by angles (degrees) |
| `interpTraj` | `(trajs, num_points, mode)` | Interpolate to num_points |
| `geometricDistance` | `(pred, gt, reduction)` | Haversine distance in meters |
| `computeJSD` | `(dataset1, dataset2, num_bins)` | Jensen-Shannon Divergence |
| `plotTraj` | `(ax, trajs, lengths, color)` | Plot on matplotlib axis |

### Example: Preprocessing Pipeline

```python
def __getitem__(self, idx):
    traj = self.trajectories[idx]
    
    # Center and normalize
    traj = centerTraj(traj)
    traj = zScoreTraj(traj)
    
    # Augmentation
    if self.augment:
        angle = torch.rand(1) * 360
        traj = rotateTraj(traj, angle)
    
    # Interpolate to fixed length
    traj = interpTraj(traj, num_points=128, mode='linear')
    
    return {'traj': traj}
```

## Notes

- **Preloading Philosophy**: Load all data to GPU tensors in `__init__` for zero overhead during training
- **Dictionary Return**: Enables flexible model interfaces
- **Set Name Convention**: train/eval/test/debug/all standard across all datasets
- **DEVICE Variable**: Imported from `JimmyTorch.Datasets`, automatically set to cuda/cpu
