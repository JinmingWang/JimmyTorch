"""One post-training, batch-one forward profile for the experiment dashboard."""
from contextlib import closing
from copy import copy, deepcopy
import json
import logging
import sqlite3
from pathlib import Path

from . import storage
from .constants import RUN_DB_FILENAME

log = logging.getLogger(__name__)


def _profile_forward(model_cfg, trained, eval_set):
    import torch
    from calflops.calculate_pipline import CalFlopsPipline, module_flop_count, module_mac_count

    # The profiler installs hooks and changes module attributes. Rebuild an eager
    # CPU model with the trained weights, leaving the returned trainer untouched.
    model = deepcopy(model_cfg).build().cpu()
    model.load_state_dict(trained.state_dict())
    model.eval()
    dataset = copy(eval_set)
    dataset.batch_size = 1
    dataset.shuffle = False
    batch = next(iter(dataset))
    args, kwargs = model.getProfileInputs(batch)

    shapes = []

    def cpu(value):
        if isinstance(value, torch.Tensor):
            shapes.append(list(value.shape))
            return value.detach().to(device="cpu", copy=True)
        if isinstance(value, tuple):
            return tuple(cpu(v) for v in value)
        if isinstance(value, list):
            return [cpu(v) for v in value]
        if isinstance(value, dict):
            return {k: cpu(v) for k, v in value.items()}
        return value

    args, kwargs = cpu(args), cpu(kwargs)
    # calculate_flops() does not remove its process-wide patches if forward
    # raises. Own the same profiler lifecycle so custom model errors are safe.
    profiler = CalFlopsPipline(model, include_backPropagation=False, compute_bp_factor=0)
    flop_depth, mac_depth = len(module_flop_count), len(module_mac_count)
    profiler.start_flops_calculate()
    try:
        model(*args, **kwargs)
        return {"forward_macs": profiler.get_total_macs(), "forward_flops": profiler.get_total_flops(),
                "batch_size": 1, "input_shapes": shapes, "device": "cpu", "profiler": "calflops"}
    finally:
        profiler.end_flops_calculate()
        del module_flop_count[flop_depth:]
        del module_mac_count[mac_depth:]


def record_model_profile(model_cfg, trained, eval_set, log_dir, save_dir):
    """Best effort after successful training. The regular run logger is closed."""
    import torch

    profile = {}
    try:
        profile["parameters"] = sum(p.numel() for p in trained.parameters())
        for filename in ("last.pth", "best.pth"):
            path = Path(save_dir) / filename
            if path.is_file():
                profile["checkpoint_mib"] = path.stat().st_size / (1024 ** 2)
                break
        cuda_devices = {p.device.index for p in trained.parameters() if p.is_cuda}
        with torch.random.fork_rng(devices=list(cuda_devices)), torch.no_grad():
            profile.update(_profile_forward(model_cfg, trained, eval_set))
    except Exception as error:
        profile["error"] = f"{type(error).__name__}: {error}"
        log.warning("Forward profile unavailable: %s", profile["error"])
    try:
        # A run deleted during profiling must stay deleted.
        db = (Path(log_dir) / RUN_DB_FILENAME).resolve()
        with closing(sqlite3.connect(db.as_uri() + "?mode=rw", uri=True)) as conn:
            storage.set_meta(conn, "model_profile", json.dumps(profile, allow_nan=False))
    except Exception as error:
        log.warning("Could not save model profile: %s", error)
