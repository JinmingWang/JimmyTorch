"""Opt-in DDP model adapter; the legacy JimmyModel remains untouched."""
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from Models.JimmyModel import JimmyModel


class ForwardBridge(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        # Bypass only the routing override, retaining nn.Module hooks/compilation.
        return nn.Module.__call__(self.model, *args, **kwargs)


class DistributedJimmyModel(JimmyModel):
    """Retains trainStep/backwardOptimize for one optimizer update per batch.

    Custom trainStep must use self(...) and self.backwardOptimize(loss).
    Overrides must follow identical forward/backward control flow on every rank.
    """
    def initialize(self):
        # Reuse optimizer creation without compiling the DDP dispatch machinery.
        compile_model = self.compile_model
        self.compile_model = False
        try:
            super().initialize()
        finally:
            self.compile_model = compile_model
        if compile_model and not getattr(self, "_forward_compiled", False):
            torch.set_float32_matmul_precision("high")
            self.forward = torch.compile(self.forward)
            self._forward_compiled = True

    def enableDistributed(self, context):
        if self.__dict__.get("_distributed_forward") is not None:
            raise RuntimeError("Model already attached to a distributed lifecycle")
        if any(parameter.device != context.device for parameter in self.parameters()):
            raise ValueError("Move model to context.device before initialize/enableDistributed")
        distributed_forward = DistributedDataParallel(
            ForwardBridge(self),
            device_ids=[context.local_rank] if context.device.type == "cuda" else None,
        )
        # Registering this reference would create a module cycle and alter checkpoints.
        object.__setattr__(self, "_distributed_forward", distributed_forward)
        object.__setattr__(self, "_distributed_context", context)
        object.__setattr__(self, "_original_backward", self.backwardOptimize)
        object.__setattr__(self, "_instance_backward", self.__dict__.get("backwardOptimize"))
        object.__setattr__(self, "backwardOptimize", self._guardedBackward)
        self._step_active = False
        self._backward_count = 0
        self._forward_count = 0

    def disableDistributed(self):
        if self.__dict__.get("_distributed_forward") is None:
            return
        original_instance = self.__dict__.pop("_instance_backward")
        if original_instance is None:
            self.__dict__.pop("backwardOptimize", None)
        else:
            object.__setattr__(self, "backwardOptimize", original_instance)
        for name in ("_distributed_forward", "_distributed_context", "_original_backward"):
            self.__dict__.pop(name, None)
        self._step_active = False

    def __call__(self, *args, **kwargs):
        distributed_forward = self.__dict__.get("_distributed_forward")
        if distributed_forward is not None and self.training and torch.is_grad_enabled():
            self._forward_count += 1
            return distributed_forward(*args, **kwargs)
        return super().__call__(*args, **kwargs)

    def beginStep(self):
        self._step_active = True
        self._backward_count = 0
        self._forward_count = 0

    def endStep(self):
        self._step_active = False
        if self._backward_count != 1 or self._forward_count < 1:
            raise RuntimeError("trainStep must call self(...) and backwardOptimize(loss) once per batch")

    def _guardedBackward(self, loss):
        def validateBackward():
            if not self._step_active or self._backward_count or not self._forward_count:
                raise RuntimeError("Distributed training requires one backwardOptimize call per batch after self(...)")
            if not isinstance(loss, torch.Tensor) or loss.numel() != 1 or not loss.requires_grad:
                raise ValueError("backwardOptimize requires one differentiable scalar loss")
        self._distributed_context.runAll(validateBackward)
        self._distributed_context.checkFinite(loss)
        self._backward_count += 1
        return self._original_backward(loss)

    def backwardOptimize(self, loss):
        if not torch.isfinite(loss.detach()).all():
            raise FloatingPointError("Non-finite training loss")
        if self.mixed_precision:
            self.scaler.scale(loss).backward()
            if self.clip_grad > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if self.clip_grad > 0:
                nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad)
            self.optimizer.step()
        self.optimizer.zero_grad()

    def loadFrom(self, path):
        device = next(self.parameters()).device
        state_dict = torch.load(path, map_location=device, weights_only=True)
        # Distributed runs require a complete, unambiguous initial model state.
        self.load_state_dict(state_dict)
