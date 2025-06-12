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
                 compile_model: bool = False,
                 clip_grad: float = 0.0):
        super(JimmyModel, self).__init__()
        self.train_loss_names = ["Train_loss"]
        self.eval_loss_names = ["Eval_loss"]
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer_cls = optimizer_cls
        self.optimizer_args = optimizer_args
        self.mixed_precision = mixed_precision
        self.compile_model = compile_model
        try:
            self.scaler = torch.amp.GradScaler() if mixed_precision else None
            # torch.amp.GradScaler may be torch.cuda.amp.GradScaler in some versions
        except AttributeError:
            self.scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
        self.clip_grad = clip_grad



    def initialize(self) -> None:
        """
        Initialize the optimizer for the model.
        :param optimizer_cls: The optimizer class to use (e.g., torch.optim.Adam).
        :param optimizer_args: A dictionary of arguments to pass to the optimizer constructor.
        :return:
        """
        if self.compile_model:
            torch.set_float32_matmul_precision('high')
            self.compile()

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
                loss = self.loss_fn(output, data_dict['target'])

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
            loss = self.loss_fn(output, data_dict['target'])
            loss.backward()

            # Standard gradient clipping (if specified)
            if self.clip_grad > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)

            # Optimizer step
            self.optimizer.step()

        # Zero the gradients
        self.optimizer.zero_grad()

        return {"Train_loss": loss.item()}, {"output": output.detach()}

    def evalStep(self, data_dict) -> (dict[str, Any], dict[str, Any]):
        with torch.no_grad():
            output = self(data_dict['data']).detach()
            loss = self.loss_fn(output, data_dict['target']).item()
        return {"Eval_loss": loss}, {"output": output.detach()}


    def saveTo(self, path: str):
        torch.save(self.state_dict(), path)


    def loadFrom(self, path: str):
        state_dict = torch.load(path, weights_only=False)
        current_state_dict = self.state_dict()

        # Filter and handle mismatched parameters
        mis_matched_keys = set()
        loadable_state_dict = {}
        for param_name, param_value in state_dict.items():
            if param_name in current_state_dict:
                if current_state_dict[param_name].size() == param_value.size():
                    loadable_state_dict[param_name] = param_value
                else:
                    mis_matched_keys.add(param_name)
                    print(
                        f"Warning! Parameter '{param_name}' expect size {current_state_dict[param_name].shape} but got {param_value.shape}. Skipping.")
            else:
                print(f"Unexpected parameter '{param_name}''. Skipping.")

        # Load filtered parameters
        self.load_state_dict(loadable_state_dict, strict=False)

        # Check for missing parameters
        for param_name in current_state_dict.keys():
            if param_name not in loadable_state_dict and param_name not in mis_matched_keys:
                print(f"Missing parameter '{param_name}' in model '{self.__class__.__name__}'.")
