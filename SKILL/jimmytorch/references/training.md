# Training & Experiment Pipeline Guide

Source code in `JimmyTorch/Training/`

## Experiment Setup

This is the `main_*.py` entry script template.

```python
from TrainEvalTest.Experiment import Experiment

def train(comments: str, dir_name: str, size: str):
    experiment = Experiment(comments, dir_name=dir_name)
    
    # Configure dataset
    experiment.train_set_cfg.cls = MyDataset
    experiment.train_set_cfg.add(batch_size=64, shuffle=True)
    
    experiment.eval_set_cfg.cls = MyDataset
    experiment.eval_set_cfg.batch_size = 256
    
    # Configure model
    size_configs = {
        "small": {"d_model": 128, "n_layers": 4},
        "medium": {"d_model": 256, "n_layers": 6},
        "large": {"d_model": 512, "n_layers": 8},
    }
    
    experiment.model_cfg.cls = MyModel
    experiment.model_cfg.add(**size_configs[size])
    experiment.model_cfg.add(
        mixed_precision=True,
        compile_model=True,
        clip_grad=1.0,
    )
    
    # Configure optimizer
    experiment.optimizer_cfg.lr = 2e-4
    experiment.optimizer_cfg.weight_decay = 1e-4
    
    # Configure LR scheduler
    experiment.lr_scheduler_cfg.mode = "min"  # Minimize metric
    experiment.lr_scheduler_cfg.patience = 10
    experiment.lr_scheduler_cfg.factor = 0.5
    
    # Training constants
    experiment.constants["n_epochs"] = 100
    experiment.constants["eval_interval"] = 5
    experiment.constants["moving_avg"] = 1000
    experiment.constants["early_stop_lr"] = 1e-6
    
    # Start training
    trainer = experiment.start()
    # Or resume: experiment.start("Runs/.../last.pth")
    
    return trainer

if __name__ == "__main__":
    train("Experiment description here", dir_name="exp1", size="medium")
```

## DynamicConfig System

Defers object instantiation for runtime modification:

```python
# Define config
cfg = DynamicConfig(cls=MyModel)
cfg.add(d_model=256, n_layers=4)

# Modify before building
cfg.add(n_layers=6)  # Update
cfg.remove('d_model')  # Remove

# Build instance
model = cfg.build()
```

## Trainer and Experiment class

- `Trainer`: Executes training loop, handles evaluation, checkpointing, LR scheduling, a template is at `JimmyTorch/JimmyTrainer.py`.
- `Experiment`: Manages overall experiment, holds configs, starts training, a template is at `JimmyTorch/JimmyExperiment.py`.

## Runtime Parameter Hot-Reload

Edit `runtime_param_buffer.yaml` during training:

```yaml
# Runs/.../runtime_param_buffer.yaml
LR: 0.0001  # Change this to adjust learning rate
```

Changes apply on next epoch. No need to restart training!

## Resume Training

```python
trainer = experiment.start("Runs/.../last.pth")
```

## Early Stopping

Stop when learning rate drops below threshold:

```python
experiment.constants["early_stop_lr"] = 1e-6
```

Prevents wasting compute on converged models.

## Testing & Evaluation

Separate from training for flexibility:

```python
def test(run_folder: str, test_city: str):
    # Load model
    model = MyModel(...).to(DEVICE)
    model.loadFrom(f"{run_folder}/best.pth")
    
    # Create experiment
    experiment = Experiment("Testing")
    experiment.test_set_cfg.cls = MyDataset
    experiment.test_set_cfg.add(city=test_city)
    test_set = experiment.test_set_cfg.build()
    
    # Test and get report
    report_df = experiment.test(model, test_set)
    report_df.to_csv(f"{run_folder}/test_{test_city}.csv")
    
    return report_df
```

The `test()` method returns a pandas DataFrame with per-sample metrics.

## Progress Monitoring

1. Terminal display with Rich library
2. The `log.txt` file for LLM-readable logs in the train folder
3. TensorBoard for detailed metric visualization
