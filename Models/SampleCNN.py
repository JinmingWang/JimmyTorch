from .JimmyModel import JimmyModel, Any
from .Basics import Conv2DBnLeakyReLU, FCLayers
import torch.nn as nn
import torch


class SampleCNN(JimmyModel):
    """
    SampleCNN is a simple CNN model for demonstration purposes.
    """

    def __init__(self, *args, **kwards):
        super(SampleCNN, self).__init__(*args, **kwards)
        self.conv1 = Conv2DBnLeakyReLU(1, 8, k=5, s=1, p=2)
        self.conv2 = Conv2DBnLeakyReLU(8, 16, k=3, s=1, p=1)

        self.conv3 = Conv2DBnLeakyReLU(16, 32, k=3, s=1, p=1)
        self.conv4 = Conv2DBnLeakyReLU(32, 64, k=3, s=1, p=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.flatten = nn.Flatten()

        self.fc = FCLayers([64 * 7 * 7, 256, 10], act=nn.LeakyReLU(inplace=True, negative_slope=0.01))

        self.loss_names = ["CE_Loss"]
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.pool(self.conv2(self.conv1(x)))
        x = self.pool(self.conv4(self.conv3(x)))
        x = self.flatten(x)
        return self.fc(x)


    def trainStep(self, data_dict) -> (dict[str, Any], dict[str, Any]):
        if self.mixed_precision:
            # Automatic Mixed Precision (AMP) forward pass and loss calculation
            with torch.autocast(device_type=data_dict['input'].device, dtype=torch.float16):
                output = self(data_dict['input'])
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
            output = self(data_dict['input'])
            loss = self.loss_fn(output, data_dict['target'])
            loss.backward()

            # Standard gradient clipping (if specified)
            if self.clip_grad > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)

            # Optimizer step
            self.optimizer.step()

        # Zero the gradients
        self.optimizer.zero_grad()

        return {"CE_Loss": loss.item()}, {"output": output.detach()}
