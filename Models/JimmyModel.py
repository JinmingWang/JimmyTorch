import torch
import torch.nn as nn
from typing import Any


class JimmyModel(nn.Module):
    """
    JimmyModel defines a model format that compatible with many different models.

    Most importantly, the forwardBackward function returns a dictionary, this is helpful for unifying the training code
    when you want to use different models for different datasets in your experiments.
    """

    def __init__(self,
                 optimizer_cls=None,
                 optimizer_args=None,
                 mixed_precision: bool = False,
                 clip_grad: float = 0.0):
        super(JimmyModel, self).__init__()
        self.train_loss_names = ["loss"]
        self.train_loss_fn = nn.CrossEntropyLoss()
        self.optimizer_cls = optimizer_cls
        self.optimizer_args = optimizer_args
        self.mixed_precision = mixed_precision
        try:
            self.scaler = torch.amp.GradScaler() if mixed_precision else None
            # torch.amp.GradScaler may be torch.cuda.amp.GradScaler in some versions
        except AttributeError:
            self.scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
        self.clip_grad = clip_grad



    def initOptimizer(self) -> None:
        """
        Initialize the optimizer for the model.
        :param optimizer_cls: The optimizer class to use (e.g., torch.optim.Adam).
        :param optimizer_args: A dictionary of arguments to pass to the optimizer constructor.
        :return:
        """
        if self.optimizer_cls is None or self.optimizer_args is None:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        else:
            self.optimizer = self.optimizer_cls(self.parameters(), **self.optimizer_args)


    @property
    def lr(self):
        return self.optimizer.param_groups[0]['lr']


    def trainStep(self, data_dict) -> (dict[str, Any], dict[str, Any]):
        if self.mixed_precision:
            # Automatic Mixed Precision (AMP) forward pass and loss calculation
            with torch.autocast(device_type=data_dict['data'].device, dtype=torch.float16):
                output = self(data_dict['data'])
                loss = self.train_loss_fn(output, data_dict['target'])

            # Backward pass with AMP
            self.scaler.scale(loss).backward()

            # Gradient clipping with AMP (if specified)
            if self.clip_grad > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)

            # Optimizer step with AMP
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard forward pass and backward pass
            output = self(data_dict['data'])
            loss = self.train_loss_fn(output, data_dict['target'])
            loss.backward()

            # Standard gradient clipping (if specified)
            if self.clip_grad > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)

            # Optimizer step
            self.optimizer.step()

        # Zero the gradients
        self.optimizer.zero_grad()

        return {"loss": loss.item()}, {"output": output.detach()}
