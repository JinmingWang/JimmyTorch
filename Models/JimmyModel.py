import torch
import torch.nn as nn
from typing import Any
from contextlib import nullcontext


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
        self.train_loss_names = ["Train/Main"]
        self.eval_loss_names = ["Eval/Main"]
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer_cls = optimizer_cls
        self.optimizer_args = optimizer_args
        self.mixed_precision = mixed_precision
        self.compile_model = compile_model
        try:
            self.scaler = torch.amp.GradScaler(init_scale=2.0**14) if mixed_precision else None
            # torch.amp.GradScaler may be torch.cuda.amp.GradScaler in some versions
        except AttributeError:
            self.scaler = torch.cuda.amp.GradScaler(init_scale=2.0**14) if mixed_precision else None
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


    def backwardOptimize(self, loss):
        """ Perform backward pass and optimizer step """
        # if loss is NaN, skip this step
        if torch.isnan(loss):
            print("NaN loss!")
            # 发出3秒钟440Hz的声音
            import os
            os.system("play -nq -t alsa synth 3 sine 440")
            return
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


    def trainStep(self, data_dict) -> (dict[str, Any], dict[str, Any]):

        with torch.autocast(device_type=data_dict['data'].device, dtype=torch.float16) if self.mixed_precision else nullcontext():
            output = self(data_dict['data'])
            loss = self.loss_fn(output, data_dict['target'])

        self.backwardOptimize(loss)

        return {"Train/Main": loss.item()}, {"output": output.detach()}

    def evalStep(self, data_dict) -> (dict[str, Any], dict[str, Any]):
        with torch.no_grad():
            output = self(data_dict['data']).detach()
            loss = self.loss_fn(output, data_dict['target']).item()
        return {"Eval/Main": loss}, {"output": output.detach()}


    def testStep(self, data_dict) -> (dict[str, Any], dict[str, Any]):
        # Sometimes, test steps can be different from eval steps.
        # For example, it may not want to draw any figures, and want to return unreduced losses
        return self.evalStep(data_dict)


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
