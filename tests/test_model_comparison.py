import json
import sqlite3

import torch

from Training.ExperimentManager import ExperimentLogger, RUN_DB_FILENAME
from Training.ExperimentManager.comparison import comparison_rows


def test_comparison_summarizes_retained_steps_and_numeric_settings(tmp_path):
    run = tmp_path / "data" / "model" / "run"
    logger = ExperimentLogger(str(run), "data", "model", "run")
    logger.log_parameter_count(torch.nn.Linear(2, 3))
    logger.log_hparams({
        "dataset": "-- Dataset Config --\nbatch_size: 32\nshuffle: True",
        "model": "Config of M: {'cls': <class 'module.Model'>, 'optimizer_args': {'lr': 1e-5}}",
        "custom": {"width": 12}, "custom.width": 99, "note": "training",
    })
    logger.close()
    with sqlite3.connect(run / RUN_DB_FILENAME) as conn:
        conn.executemany("INSERT INTO scalars(tag,step,wall_time,value) VALUES (?,?,?,?)", [
            ("Eval/custom", 20, 15, -1), ("Eval/custom", 5, 10, 100),
            ("Eval/custom", 20, 16, 6), ("Eval/custom", 21, 17, float("inf")),
            ("LR", 10, 20, 1e-5), ("LR", 40, 25, 1e-5),
        ])
    (run / "best.pth").write_bytes(b"123")
    (run / "last.pth").write_bytes(b"1234")
    (run / "model_profile.json").write_text(json.dumps({"forward_macs": 7, "forward_flops": 14}))
    rows = comparison_rows(tmp_path, {"data": {"model": ["run"]}})
    assert len(rows) == 1
    row = rows[0]
    assert row["metrics"]["Eval/custom"] == {"lowest": -1, "highest": 100, "last": 6}
    assert row["hparams"] == {"custom.width": 12, r"custom\.width": 99,
                              "dataset.batch_size": 32, "model.optimizer_args.lr": 1e-5}
    assert row["measurements"] == {"checkpoint_mib": 4 / 1024**2, "iterations_per_second": 6,
                                   "parameters": 9, "forward_macs": 7, "forward_flops": 14}


def test_legacy_partial_and_uninitialized_runs(tmp_path):
    base = tmp_path / "dataset" / "model"
    run = base / "legacy"
    run.mkdir(parents=True)
    (run / "best.pth").write_bytes(b"123")
    (run / "model_profile.json").write_text('{"forward_macs":null,"forward_flops":-2,"parameters":0}')
    partial = base / "new"
    logger = ExperimentLogger(str(partial), "dataset", "model", "new")
    logger.log_parameter_count(torch.nn.LazyLinear(2))
    logger.log_scalar("LR", 1, 0.0)
    logger.close()
    (partial / "model_profile.json").write_text("{")
    legacy, new = comparison_rows(tmp_path, {"dataset": {"model": ["legacy", "new"]}})
    assert legacy["measurements"] == {"checkpoint_mib": 3 / 1024**2, "parameters": 0}
    assert not (run / RUN_DB_FILENAME).exists()
    assert new["measurements"] == {}
    assert new["metrics"]["LR"] == {"lowest": 0, "highest": 0, "last": 0}


def test_profile_uses_one_example_and_restores_profiler_after_forward_error(tmp_path):
    from DynamicConfig import DynamicConfig
    from Models.JimmyModel import JimmyModel
    from Training.ExperimentManager.model_profile import record_model_profile

    class Sample:
        batch_size = 16
        shuffle = True

        def __iter__(self):
            yield {"features": torch.ones(self.batch_size, 3), "other": torch.ones(self.batch_size, 2)}

    class Model(JimmyModel):
        def __init__(self, fail=False):
            super().__init__()
            self.layer = torch.nn.Linear(3, 2)
            self.fail = fail

        def getProfileInputs(self, batch):
            return (batch["features"],), {"other": batch["other"]}

        def forward(self, features, *, other):
            assert not self.training and not torch.is_grad_enabled()
            assert features.shape == (1, 3) and other.shape == (1, 2)
            result = self.layer(features)
            if self.fail:
                raise RuntimeError("unsupported forward")
            return result + other

    run = tmp_path / "data" / "model" / "run"
    save = tmp_path / "weights"
    save.mkdir()
    (save / "last.pth").write_bytes(b"weights")
    logger = ExperimentLogger(str(run), "data", "model", "run")
    logger.log_scalars(4, LR=0.001)
    logger.close()
    functional_linear = torch.nn.functional.linear
    dataset = Sample()
    for fail in (True, False):
        cfg = DynamicConfig(Model, fail=fail)
        trained = cfg.build()
        state = torch.random.get_rng_state()
        record_model_profile(cfg, trained, dataset, run, save)
        with sqlite3.connect(run / RUN_DB_FILENAME) as conn:
            meta = dict(conn.execute("SELECT key,value FROM meta"))
        profile = json.loads(meta["model_profile"])
        assert meta["status"] == "done"
        assert trained.training and dataset.batch_size == 16 and dataset.shuffle
        assert functional_linear is torch.nn.functional.linear
        assert torch.equal(state, torch.random.get_rng_state())
        if fail:
            assert "unsupported forward" in profile["error"]
            assert "forward_macs" not in profile
        else:
            assert profile["batch_size"] == 1 and profile["input_shapes"] == [[1, 3], [1, 2]]
            assert profile["forward_macs"] == 6 and profile["forward_flops"] >= 12
            values = comparison_rows(tmp_path, {"data": {"model": ["run"]}})[0]["measurements"]
            assert values["forward_macs"] == 6 and values["parameters"] == 8
            assert values["checkpoint_mib"] == 7 / 1024**2 and values["iterations_per_second"] > 0
