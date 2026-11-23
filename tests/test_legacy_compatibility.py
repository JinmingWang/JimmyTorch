"""Legacy APIs stay independent of the opt-in distributed lifecycle."""
import inspect
import os
from pathlib import Path
import subprocess
import sys

import torch
from Models.JimmyModel import JimmyModel
from Datasets.JimmyDataset import JimmyDataset
from JimmyExperiment import JimmyExperiment
from JimmyTrainer import JimmyTrainer


class LegacyRegressor(JimmyModel):
    def __init__(self):
        super().__init__(optimizer_cls=torch.optim.SGD, optimizer_args={"lr": 0.1})
        self.linear = torch.nn.Linear(1, 1)
        self.loss_fn = torch.nn.MSELoss()
        self.optimizations = 0
        self.steps = 0

    def forward(self, data):
        return self.linear(data)

    def trainStep(self, data_dict):
        self.steps += 1
        output = self(data_dict["data"])
        loss = self.loss_fn(output, data_dict["target"])
        self.backwardOptimize(loss)
        return {"Train/Main": loss.item()}, {"output": output.detach()}

    def backwardOptimize(self, loss):
        self.optimizations += 1
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def evalStep(self, data_dict):
        return self.testStep(data_dict)


class LegacyDataset(JimmyDataset):
    def __init__(self, set_name, batch_size=2):
        super().__init__(set_name, batch_size)
        self.n_samples = 4

    def __getitem__(self, idx):
        indices = self._indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        data = indices.float().reshape(-1, 1)
        return {"data": data, "target": data * 2}


def test_legacySignaturesAndImports():
    assert "distributed_context" not in inspect.signature(JimmyTrainer).parameters
    assert not hasattr(JimmyModel, "computeTrainLoss")
    assert not hasattr(JimmyDataset, "configureDistributed")
    assert list(inspect.signature(JimmyExperiment.start).parameters) == ["self", "checkpoint"]
    repo_dir = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-c", "import JimmyExperiment, sys; assert 'Distributed' not in sys.modules"],
        cwd=repo_dir, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_legacyCustomHooksAndCheckpoint(tmp_path):
    model = LegacyRegressor()
    model.initialize()
    before = [parameter.detach().clone() for parameter in model.parameters()]
    for batch in LegacyDataset("train"):
        model.trainStep(batch)
    assert model.steps == model.optimizations == 2
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))
    checkpoint = tmp_path / "legacy.pth"
    model.saveTo(str(checkpoint))
    restored = LegacyRegressor()
    restored.loadFrom(str(checkpoint))
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name])


def test_legacyTrainerKeepsCustomSteps(tmp_path, monkeypatch):
    import JimmyTrainer as trainer_module
    from Training.ExperimentManager.client import ExperimentManagerClient
    from collections import deque

    class CpuAverage:
        # Legacy MovingAvg hardcodes CUDA. Substitute only its storage for CPU CI.
        def __init__(self, window_size):
            self.values = deque(maxlen=window_size)
        def update(self, value):
            self.values.append(value)
        def get(self):
            return sum(self.values) / len(self.values)

    monkeypatch.setattr(trainer_module, "MovingAvg", CpuAverage)
    monkeypatch.setattr(ExperimentManagerClient, "_register_and_start", lambda self: None)
    monkeypatch.setattr(ExperimentManagerClient, "_post_silently", lambda *args: None)
    model = LegacyRegressor()
    model.initialize()
    trainer = JimmyTrainer(LegacyDataset("train"), LegacyDataset("eval"), model,
        torch.optim.lr_scheduler.StepLR(model.optimizer, step_size=1), str(tmp_path), str(tmp_path),
        n_epochs=2, moving_avg=1)
    trainer.start()
    assert model.steps == model.optimizations == 4
    assert (tmp_path / "best.pth").exists()
    assert (tmp_path / "last.pth").exists()
