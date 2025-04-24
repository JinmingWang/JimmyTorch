from ..JimmyModel import JimmyModel, Any
from ..Basics import Conv2DBnLeakyReLU, FCLayers
import torch.nn as nn
import torch
from .components import Block


class SampleCNN(JimmyModel):
    """
    SampleCNN is a simple CNN model for demonstration purposes.
    """

    def __init__(self, *args, **kwards):
        super(SampleCNN, self).__init__(*args, **kwards)
        self.b1 = Block(1, 16)

        self.b2 = Block(16, 64)

        self.flatten = nn.Flatten()

        self.fc = FCLayers([64 * 7 * 7, 256, 10], act=nn.LeakyReLU(inplace=True, negative_slope=0.01))

        self.train_loss_names = ["Train_CE"]
        self.eval_loss_names = ["Eval_CE"]
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.flatten(x)
        return self.fc(x)


    def trainStep(self, data_dict) -> (dict[str, Any], dict[str, Any]):
        if self.mixed_precision:
            # Automatic Mixed Precision (AMP) forward pass and loss calculation
            with torch.autocast(device_type=data_dict['input'].device, dtype=torch.float16):
                output = self(data_dict['input'])
                loss = self.ce_loss(output, data_dict['target'])

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
            output = self(data_dict['input'])
            loss = self.ce_loss(output, data_dict['target'])
            loss.backward()

            # Standard gradient clipping (if specified)
            if self.clip_grad > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)

            # Optimizer step
            self.optimizer.step()

        # Zero the gradients
        self.optimizer.zero_grad()

        return {"Train_CE": loss.item()}, {"output": output.detach()}


    def evalStep(self, data_dict) -> (dict[str, Any], dict[str, Any]):
        with torch.no_grad():
            output = self(data_dict['input']).detach()
            loss = self.ce_loss(output, data_dict['target']).item()
        return {"Eval_CE": loss}, {"output": output.detach()}