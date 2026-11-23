"""Fixed-membership multi-machine distributed execution, independent of dashboard threads."""
from datetime import timedelta
import os
import socket

import torch
import torch.distributed as dist


class DistributedContext:
    def __init__(self, backend=None, timeout_seconds=1800):
        required = ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
        if any(name not in os.environ for name in required):
            raise RuntimeError("DistributedJimmyExperiment requires torchrun (even for one worker).")
        self.rank = int(os.environ["RANK"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        if self.world_size < 1 or self.local_world_size < 1 or self.world_size % self.local_world_size:
            raise ValueError("Use the same positive worker count on every machine.")
        if not 0 <= self.rank < self.world_size or not 0 <= self.local_rank < self.local_world_size:
            raise ValueError("Invalid torchrun worker ranks.")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        self.node_rank = int(os.environ.get("GROUP_RANK", self.rank // self.local_world_size))
        self.node_id = str(self.node_rank)
        self.hostname = socket.gethostname()
        self.is_distributed = True
        self.is_primary = self.rank == 0
        self.is_node_primary = self.local_rank == 0
        self.owns_group = False
        use_cuda = torch.cuda.is_available() and backend != "gloo"
        self.device = torch.device("cuda", self.local_rank) if use_cuda else torch.device("cpu")
        if use_cuda:
            if self.local_rank >= torch.cuda.device_count():
                raise ValueError("Launch at most one worker per visible GPU.")
            torch.cuda.set_device(self.device)
        self.backend = backend or ("nccl" if use_cuda else "gloo")
        if self.is_distributed:
            if dist.is_initialized():
                if dist.get_rank() != self.rank or dist.get_world_size() != self.world_size:
                    raise ValueError("Existing process group does not match torchrun environment.")
            else:
                dist.init_process_group(self.backend, timeout=timedelta(seconds=timeout_seconds))
                self.owns_group = True

    def broadcastObject(self, value):
        values = [value if self.is_primary else None]
        if self.is_distributed:
            dist.broadcast_object_list(values, src=0)
        return values[0]

    def gatherObjects(self, value):
        values = [None] * self.world_size
        if self.is_distributed:
            dist.all_gather_object(values, value)
        else:
            values[0] = value
        return values

    def runPrimary(self, action):
        """Propagate primary-only I/O/evaluation errors to waiting workers."""
        result, error_message = None, None
        if self.is_primary:
            try:
                result = action()
            except Exception as error:
                error_message = f"{type(error).__name__}: {error}"
        result, error_message = self.broadcastObject((result, error_message))
        if error_message is not None:
            raise RuntimeError(error_message)
        return result

    def runAll(self, action):
        """Fence operations without collectives, so one worker's error reaches all."""
        result, error_message = None, None
        try:
            result = action()
        except Exception as error:
            error_message = f"Rank {self.rank}: {type(error).__name__}: {error}"
        errors = self.gatherObjects(error_message)
        if any(errors):
            raise RuntimeError("; ".join(error for error in errors if error))
        return result

    def checkFinite(self, loss):
        finite = torch.isfinite(loss.detach()).all().to(device=self.device, dtype=torch.int32)
        if self.is_distributed:
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not finite.item():
            raise FloatingPointError("Non-finite training loss on at least one rank; training stopped.")

    def averageMetrics(self, metrics, names):
        values = torch.tensor([float(metrics[name]) for name in names], device=self.device, dtype=torch.float64)
        if self.is_distributed:
            dist.all_reduce(values)
            values /= self.world_size
        return dict(zip(names, values.cpu().tolist()))

    def deviceAssignments(self):
        device_id = None
        if self.device.type == "cuda":
            # CUDA's UUID follows logical device visibility/order; NVML indices do not.
            try:
                device_id = getattr(torch.cuda.get_device_properties(self.device), "uuid", None)
                if isinstance(device_id, bytes):
                    device_id = device_id.decode("ascii")
                elif device_id is not None:
                    device_id = str(device_id)
            except Exception:
                pass  # Telemetry identity must never prevent training.
        return self.gatherObjects({"rank": self.rank, "device_id": device_id, "device_type": self.device.type,
                                   "node_id": self.node_id, "hostname": self.hostname,
                                   "local_rank": self.local_rank, "local_world_size": self.local_world_size})

    def close(self):
        if self.owns_group and dist.is_initialized():
            dist.destroy_process_group()
            self.owns_group = False


def moveToDevice(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: moveToDevice(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        items = [moveToDevice(item, device) for item in value]
        return type(value)(*items) if hasattr(value, "_fields") else tuple(items)
    if isinstance(value, list):
        return [moveToDevice(item, device) for item in value]
    return value
