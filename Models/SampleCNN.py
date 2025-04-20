from .JimmyModel import JimmyModel
from .Basics import Conv2DBnLeakyReLU, FCLayers
import torch.nn as nn
import torch


class SampleCNN(JimmyModel):
    """
    SampleCNN is a simple CNN model for demonstration purposes.
    """

    def __init__(self):
        super(SampleCNN, self).__init__()
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


    def forwardBackward(self, data_dict: dict, loss_scaler = None) -> tuple[dict, dict]:
        """
        Forward pass and backward pass of the model.
        :param data_dict: a dictionary containing the input data
        :param loss_scaler: a loss scaler for mixed precision training
        :return: a dictionary containing the loss and output
        """

        if loss_scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = self(data_dict['input'])
                loss = self.loss_fn(output, data_dict['target'])
            loss_scaler.scale(loss).backward()
        else:
            output = self(data_dict['input'])
            loss = self.loss_fn(output, data_dict['target'])
            loss.backward()

        return {"CE_Loss": loss.item()}, {"output": output.detach()}
