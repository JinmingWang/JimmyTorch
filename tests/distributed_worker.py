"""Subprocess scenarios used by test_distributed.py (no downloaded datasets)."""
import json
import os
from pathlib import Path
import sys

import torch
from Distributed.Dataset import DistributedJimmyDataset as JimmyDataset
from Distributed.Model import DistributedJimmyModel as JimmyModel
from Distributed.Context import DistributedContext, moveToDevice


class TinyDataset(JimmyDataset):
    def __init__(self, set_name="train", batch_size=2, drop_last=False, shuffle=False, preparation_dir=None):
        super().__init__(set_name, batch_size, drop_last, shuffle)
        if os.environ.get("TEST_SCENARIO") == "setup_error" and int(os.environ["RANK"]) == 1:
            raise ValueError("injected setup failure")
        self.n_samples = 8 if set_name == "train" else 3

    @classmethod
    def prepareData(cls, preparation_dir=None, **kwargs):
        if preparation_dir:
            Path(preparation_dir, f"prepared-{os.environ['RANK']}").touch()

    def __getitem__(self, idx):
        if os.environ.get("TEST_SCENARIO") == "worker_exit" and int(os.environ["RANK"]) == 1:
            os._exit(7)
        if os.environ.get("TEST_SCENARIO") == "loader_error" and int(os.environ["RANK"]) == 1:
            raise ValueError("injected loader failure")
        indices = self._indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        data = indices.float().reshape(-1, 1) / 8
        return {"data": data, "target": 2 * data, "indices": indices}


class TinyModel(JimmyModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.linear = torch.nn.Linear(1, 1)
        self.loss_fn = torch.nn.MSELoss()
        self.eval_calls = 0

    def forward(self, data):
        return self.linear(data)

    def evalStep(self, data_dict):
        assert int(os.environ.get("RANK", "0")) == 0
        if os.environ.get("TEST_SCENARIO") == "eval_error":
            raise ValueError("injected evaluation failure")
        self.eval_calls += 1
        losses, outputs = self.testStep(data_dict)
        if os.environ.get("TEST_SCENARIO") == "lifecycle":
            import matplotlib.pyplot as plt
            figure, axes = plt.subplots(figsize=(2, 2))
            axes.plot([0, 1], [0, 1])
            outputs["fig"] = figure
        return losses, outputs


class LegacyModel(TinyModel):
    def trainStep(self, data_dict):
        return super().trainStep(data_dict)


class CustomOptimizeModel(LegacyModel):
    def backwardOptimize(self, loss):
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        self.optimizer.step()
        self.optimizer.zero_grad()


def runEquivalence(scenario, output_dir):
    context = DistributedContext(backend="gloo", timeout_seconds=15)
    try:
        torch.manual_seed(7)
        model = CustomOptimizeModel(optimizer_cls=torch.optim.SGD, optimizer_args={"lr": 0.1})
        if scenario == "compile":
            original_compile = torch.compile
            torch.compile = lambda method: original_compile(method, backend="aot_eager")
            model.compile_model = True
        model.initialize()
        # Weights written by one unwrapped model load on every DDP worker.
        checkpoint_path = str(Path(output_dir) / "single-weights.pth")
        context.runPrimary(lambda: model.saveTo(checkpoint_path))
        model.loadFrom(checkpoint_path)
        reference_model = CustomOptimizeModel(optimizer_cls=torch.optim.SGD, optimizer_args={"lr": 0.1})
        reference_model.load_state_dict(model.state_dict())
        reference_model.initialize()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(context.rank)  # Constructor must synchronize differing initial weights.
        model.enableDistributed(context)
        hook_calls = []
        hook_handle = model.register_forward_hook(lambda *args: hook_calls.append(1))
        for step_idx in range(3):
            global_data = torch.arange(8).float().reshape(-1, 1) / 8 + step_idx / 10
            local_data = global_data[context.rank::context.world_size]
            model.beginStep()
            target = local_data * (float("nan") if scenario == "nan" and context.rank == 1 else 2)
            metrics, _ = model.trainStep({"data": local_data, "target": target})
            model.endStep()
            reference_model.trainStep({"data": global_data, "target": global_data * 2})
            for actual, expected in zip(model.parameters(), reference_model.parameters()):
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
            mean_metrics = context.averageMetrics(metrics, ["Train/Main"])
            gathered_metrics = context.gatherObjects(mean_metrics)
            assert gathered_metrics[0] == gathered_metrics[1]
        assert len(hook_calls) == 3
        hook_handle.remove()
        assert set(model.state_dict()) == {"linear.weight", "linear.bias"}
        model.disableDistributed()
        model.enableDistributed(context)
        model.disableDistributed()
        model.trainStep({"data": global_data, "target": global_data * 2})
        print("equivalence passed", context.rank)
    finally:
        context.close()


class RecordingScheduler(torch.optim.lr_scheduler.StepLR):
    def __init__(self, *args, **kwargs):
        self.inputs = []
        super().__init__(*args, **kwargs)

    def update(self, metric):
        self.inputs.append(metric)
        self.step()


def runLifecycle(output_dir, scenario):
    import Distributed.Trainer as trainer_module
    from Distributed.Experiment import DistributedJimmyExperiment as JimmyExperiment
    from Distributed.Reporting import DistributedRunClient as ExperimentManagerClient

    class TestClient(ExperimentManagerClient):
        def _register_and_start(self):
            assert int(os.environ.get("RANK", "0")) == 0
            self.requested = False

        def consume_learning_rate_request(self):
            if not self.requested:
                self.requested = True
                return 0.04
            return None

        def _post_silently(self, path, payload):
            return None

    class TestReporter:
        def __init__(self, host, port, run_id, node_id, device_assignments, refresh_interval):
            assert int(os.environ["LOCAL_RANK"]) == 0
            assert all(entry["node_id"] == node_id for entry in device_assignments)
            Path(output_dir, f"reporter-{node_id}.json").write_text(json.dumps(device_assignments))
        def close(self):
            pass
    trainer_module.NodeReporter = TestReporter
    trainer_module.DistributedRunClient = TestClient
    torch.manual_seed(17)
    experiment = JimmyExperiment("Synthetic DDP integration", dir_name=f"rank-{os.environ.get('RANK', '0')}")
    experiment.dataset_cfg.cls = TinyDataset
    experiment.dataset_cfg.batch_size = 2
    experiment.dataset_cfg.shuffle = False
    experiment.model_cfg.cls = {"legacy": LegacyModel, "single": LegacyModel,
                                "custom": CustomOptimizeModel}.get(scenario, TinyModel)
    experiment.model_cfg.optimizer_cls = torch.optim.SGD
    experiment.model_cfg.optimizer_args = {"lr": 0.1}
    from DynamicConfig import DynamicConfig
    experiment.lr_scheduler_cfg = DynamicConfig(RecordingScheduler, step_size=1, gamma=0.5)
    experiment.constants.update(n_epochs=5, moving_avg=2, save_dir=output_dir, early_stop_lr=0.015)
    if scenario.startswith("prepare_"):
        experiment.distributed_cfg["data_prepare_scope"] = scenario.removeprefix("prepare_")
        experiment.dataset_cfg.preparation_dir = output_dir
    experiment.distributed_cfg["backend"] = "gloo"
    experiment.distributed_cfg["timeout_seconds"] = 12
    if scenario == "zero":
        experiment.dataset_cfg.batch_size = 100
        experiment.dataset_cfg.drop_last = True
    trainer = experiment.start()
    assert trainer.model.lr == 0.01
    assert trainer.model.eval_calls == (4 if int(os.environ.get("RANK", "0")) == 0 else 0)
    weights = {name: value.tolist() for name, value in trainer.model.state_dict().items()}
    Path(output_dir, f"worker-{os.environ.get('RANK', '0')}.json").write_text(json.dumps(weights))
    Path(output_dir, f"scheduler-{os.environ.get('RANK', '0')}.json").write_text(json.dumps(trainer.lr_scheduler.inputs))
    assert trainer.distributed_context.owns_group is False
    assert "_distributed_forward" not in trainer.model.__dict__
    if int(os.environ["WORLD_SIZE"]) == 2 and scenario == "lifecycle":
        torch.manual_seed(0)
        reference = TinyModel(optimizer_cls=torch.optim.SGD, optimizer_args={"lr": 0.04})
        reference.initialize()
        expected_inputs, recent_metrics = [], []
        for epoch in range(2):
            for batch_start in (0, 4):
                data = torch.arange(batch_start, batch_start + 4).float().reshape(-1, 1) / 8
                metrics, _ = reference.trainStep({"data": data, "target": data * 2})
                recent_metrics.append(metrics["Train/Main"])
            expected_inputs.append(sum(recent_metrics[-2:]) / 2)
            reference.optimizer.param_groups[0]["lr"] *= 0.5
        torch.testing.assert_close(torch.tensor(trainer.lr_scheduler.inputs), torch.tensor(expected_inputs))


def runCuda(scenario):
    from Distributed.Sample import DistributedSampleCNN as SampleCNN
    context = DistributedContext(timeout_seconds=120)
    try:
        torch.manual_seed(4)
        model = SampleCNN(mixed_precision=True, clip_grad=0.5, compile_model=scenario == "cuda_compile").to(context.device)
        model.initialize()
        model.enableDistributed(context)
        assert model.scaler.is_enabled()
        original_step = model.optimizer.step
        gradient_norms = []
        def checkedStep(*args, **kwargs):
            gradient_norms.append(torch.linalg.vector_norm(torch.stack([
                torch.linalg.vector_norm(parameter.grad.detach().float())
                for parameter in model.parameters() if parameter.grad is not None])).item())
            return original_step(*args, **kwargs)
        model.optimizer.step = checkedStep
        before = [value.detach().clone() for value in model.parameters()]
        for _ in range(2):
            batch = moveToDevice({"input": torch.randn(2, 1, 28, 28), "target": torch.tensor([1, 2])}, context.device)
            model.beginStep()
            model.trainStep(batch)
            model.endStep()
        assert gradient_norms and max(gradient_norms) <= 0.50001
        assert any(not torch.equal(left, right) for left, right in zip(before, model.parameters()))
        parameters = torch.cat([value.detach().flatten() for value in model.parameters()])
        copies = [torch.empty_like(parameters) for _ in range(context.world_size)]
        torch.distributed.all_gather(copies, parameters)
        torch.testing.assert_close(copies[0], copies[1])
        assert all(value.device == context.device for value in model.parameters())
        print("CUDA passed", context.device, context.deviceAssignments())
    finally:
        context.close()


if __name__ == "__main__":
    scenario = sys.argv[1]
    os.environ["TEST_SCENARIO"] = scenario
    if scenario in ("equivalence", "nan", "compile"):
        runEquivalence(scenario, sys.argv[2])
    elif scenario.startswith("cuda"):
        runCuda(scenario)
    else:
        runLifecycle(sys.argv[2], scenario)
