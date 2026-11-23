import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest
import torch
from Distributed.Dataset import DistributedJimmyDataset as JimmyDataset
from Distributed.Loader import DistributedLoader as MultiThreadLoader
from Models.JimmyModel import JimmyModel
from Distributed.Context import DistributedContext, moveToDevice

repo_dir = Path(__file__).resolve().parents[1]


def launchWorkers(scenario, output_dir, cpu=True, processes=2):
    env = dict(os.environ, PYTHONPATH=str(repo_dir), OMP_NUM_THREADS="1", MPLBACKEND="Agg",
               MPLCONFIGDIR=str(output_dir / "mpl"), XDG_CACHE_HOME=str(output_dir / "cache"))
    if cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    return subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone",
                           f"--nproc-per-node={processes}", str(repo_dir / "tests/distributed_worker.py"),
                           scenario, str(output_dir)], cwd=repo_dir, env=env, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180 if cpu else 600)


@pytest.mark.parametrize("sample_count,drop_last,expected_steps", [(11, False, 3), (11, True, 2), (1, False, 1)])
def test_shardingAndTails(sample_count, drop_last, expected_steps):
    rank_indices = []
    for rank in range(2):
        dataset = JimmyDataset("train", batch_size=2, drop_last=drop_last, shuffle=True)
        dataset.n_samples = sample_count
        dataset.configureDistributed(rank, 2, seed=5)
        dataset.setEpoch(0)
        indices = torch.cat([batch["indices"] for batch in dataset]).tolist()
        assert dataset.n_batches == expected_steps
        assert dataset.n_samples == sample_count
        assert torch.cat([batch["indices"] for batch in dataset]).tolist() == indices
        if sample_count > 1:
            dataset.setEpoch(1)
            assert torch.cat([batch["indices"] for batch in dataset]).tolist() != indices
        rank_indices.append(indices)
    if drop_last:
        assert not set(rank_indices[0]) & set(rank_indices[1])
    else:
        assert set(rank_indices[0] + rank_indices[1]) == set(range(sample_count))


def test_loaderExceptionAndEarlyClose():
    class BrokenDataset(JimmyDataset):
        def __getitem__(self, idx):
            raise ValueError("loader failed")
    loader = MultiThreadLoader(BrokenDataset("train", 2))
    with pytest.raises(ValueError, match="loader failed"):
        list(loader)
    assert not loader.thread.is_alive()
    loader = MultiThreadLoader(JimmyDataset("train", 1), queue_size=1)
    loader.close()
    assert not loader.thread.is_alive()


def test_recursiveDeviceTransfer():
    batch = {"nested": [torch.ones(1), (torch.zeros(2), "metadata")], "count": 3}
    moved = moveToDevice(batch, torch.device("cpu"))
    assert moved["nested"][1][1] == "metadata"
    assert moved["count"] == 3
    assert moved["nested"][0].device.type == "cpu"


def test_requiresTorchrun(monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    with pytest.raises(RuntimeError, match="requires torchrun"):
        DistributedContext()


def test_twoProcessGlobalBatchEquivalence(tmp_path):
    result = launchWorkers("equivalence", tmp_path)
    assert result.returncode == 0, result.stdout
    assert result.stdout.count("equivalence passed") == 2


def test_distributedLifecycle(tmp_path):
    result = launchWorkers("lifecycle", tmp_path)
    assert result.returncode == 0, result.stdout
    assert json.loads((tmp_path / "worker-0.json").read_text()) == json.loads((tmp_path / "worker-1.json").read_text())
    assert len(list(tmp_path.glob("*/status_and_log.sqlite"))) == 1
    weights = torch.load(tmp_path / "rank-0/last.pth", weights_only=True, map_location="cpu")
    assert weights and not any(name.startswith("module.") for name in weights)
    with sqlite3.connect(tmp_path / "rank-0/status_and_log.sqlite") as connection:
        assert connection.execute("select value from meta where key='status'").fetchone()[0] == "done"
        assert connection.execute("select count(*) from figures").fetchone()[0] == 2
    assert (tmp_path / "rank-0/best.pth").exists()
    assert json.loads((tmp_path / "scheduler-0.json").read_text()) == json.loads((tmp_path / "scheduler-1.json").read_text())
    # The DDP checkpoint is directly loadable into the original module.
    from test_legacy_compatibility import LegacyRegressor
    restored = LegacyRegressor()
    restored.loadFrom(str(tmp_path / "rank-0/last.pth"))
    for name, value in restored.state_dict().items():
        torch.testing.assert_close(value, weights[name])


@pytest.mark.parametrize("scenario,message", [("nan", "Non-finite training loss"),
    ("loader_error", "injected loader failure"), ("eval_error", "injected evaluation failure"),
    ("zero", "zero batches"), ("setup_error", "injected setup failure"), ("worker_exit", "exitcode")])
def test_workerFailuresTerminate(scenario, message, tmp_path):
    result = launchWorkers(scenario, tmp_path)
    assert result.returncode != 0
    assert message in result.stdout
    if scenario == "eval_error":
        with sqlite3.connect(tmp_path / "rank-0/status_and_log.sqlite") as connection:
            assert connection.execute("select value from meta where key='status'").fetchone()[0] == "error"


def test_singleProcessCompatibility(tmp_path):
    result = launchWorkers("single", tmp_path, processes=1)
    assert result.returncode == 0, result.stdout


@pytest.mark.parametrize("scenario", ["legacy", "custom"])
def test_originalTrainingHooksSupported(scenario, tmp_path):
    result = launchWorkers(scenario, tmp_path)
    assert result.returncode == 0, result.stdout


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two CUDA GPUs")
@pytest.mark.parametrize("scenario", ["cuda", "cuda_compile"])
def test_cudaTwoGpuTraining(scenario, tmp_path):
    result = launchWorkers(scenario, tmp_path, cpu=False)
    assert result.returncode == 0, result.stdout


def test_compiledForwardWithDdp(tmp_path):
    result = launchWorkers("compile", tmp_path)
    assert result.returncode == 0, result.stdout


def launchNodes(scenario, output_dir):
    import socket
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    env = dict(os.environ, PYTHONPATH=str(repo_dir), OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="",
               MPLBACKEND="Agg", MPLCONFIGDIR=str(output_dir / "mpl"), XDG_CACHE_HOME=str(output_dir / "cache"))
    processes = []
    handles = []
    try:
        for node_rank in range(2):
            handle = open(output_dir / f"node-{node_rank}.log", "w")
            handles.append(handle)
            processes.append(subprocess.Popen([sys.executable, "-m", "torch.distributed.run",
                "--nnodes=2", "--nproc-per-node=2", f"--node-rank={node_rank}",
                "--master-addr=127.0.0.1", f"--master-port={port}", "--max-restarts=0",
                str(repo_dir / "tests/distributed_worker.py"), scenario, str(output_dir)],
                cwd=repo_dir, env=env, stdout=handle, stderr=subprocess.STDOUT))
        for process in processes:
            process.wait(timeout=90)
        for handle in handles:
            handle.flush()
        assert all(process.returncode == 0 for process in processes), "\n".join(
            (output_dir / f"node-{index}.log").read_text() for index in range(2))
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for handle in handles:
            handle.close()


@pytest.mark.parametrize("scope", ["shared", "per_node"])
def test_twoNodeLifecycleAndPreparation(scope, tmp_path):
    launchNodes("prepare_" + scope, tmp_path)
    states = [json.loads((tmp_path / f"worker-{rank}.json").read_text()) for rank in range(4)]
    assert all(state == states[0] for state in states)
    expected_ranks = [0] if scope == "shared" else [0, 2]
    assert sorted(path.name for path in tmp_path.glob("prepared-*")) == [f"prepared-{rank}" for rank in expected_ranks]
    assert sorted(path.name for path in tmp_path.glob("reporter-*.json")) == ["reporter-0.json", "reporter-1.json"]
    assert len(list(tmp_path.glob("*/status_and_log.sqlite"))) == 1


def test_namedTupleBatchPreserved():
    from collections import namedtuple
    batch_type = namedtuple("Batch", "values label")
    batch = batch_type(torch.ones(2), "metadata")
    result = moveToDevice(batch, torch.device("cpu"))
    assert type(result) is batch_type
    assert result.label == "metadata"
    torch.testing.assert_close(result.values, batch.values)


def test_distributedMnistPreparationAndDisjointSplits(monkeypatch):
    from Distributed.Sample import DistributedMNISTDataset
    import torchvision
    calls = []
    class FakeMnist:
        def __init__(self, root, train, download):
            calls.append((train, download))
            self.data = torch.arange(100).reshape(100, 1, 1).expand(100, 28, 28)
            self.targets = torch.arange(100)
        def __len__(self):
            return 100
    monkeypatch.setattr(torchvision.datasets, "MNIST", FakeMnist)
    DistributedMNISTDataset.prepareData()
    train = DistributedMNISTDataset("train", 20)
    validation = DistributedMNISTDataset("eval", 20)
    assert calls == [(True, True), (False, True), (True, False), (True, False)]
    assert train.n_samples == 90 and validation.n_samples == 10
    assert not set(train.labels.tolist()) & set(validation.labels.tolist())
    assert train.images.device.type == "cpu"


@pytest.mark.parametrize("rank", [0, 1])
def test_examplePostTrainingRunsOnlyOnPrimary(rank, tmp_path, monkeypatch):
    import distributed_main
    from types import SimpleNamespace
    import pandas as pd
    calls = []
    class FakeExperiment:
        def __init__(self, *args):
            self.constants, self.distributed_cfg = {}, {}
        def start(self):
            return SimpleNamespace(distributed_context=SimpleNamespace(is_primary=rank == 0),
                save_dir=str(tmp_path), model=SimpleNamespace(loadFrom=lambda path: calls.append("load")))
        def test(self, model, test_set):
            calls.append("test")
            return pd.DataFrame({"Eval/Main": [1.0]})
    monkeypatch.setattr(distributed_main, "DistributedJimmyExperiment", FakeExperiment)
    monkeypatch.setattr(distributed_main, "DistributedMNISTDataset", lambda *args, **kwargs: None)
    distributed_main.main()
    assert calls == (["load", "test"] if rank == 0 else [])
    assert (tmp_path / "test_report_test.csv").exists() == (rank == 0)
