from Models.Basics import Conv2DBnLeakyReLU, FCLayers
import torch.nn as nn


class Block(nn.Sequential):
    def __init__(self, c_in: int, c_out: int):
        super().__init__(
            Conv2DBnLeakyReLU(c_in, c_out, k=3, s=1, p=1),
            Conv2DBnLeakyReLU(c_out, c_out, k=3, s=1, p=1),
            nn.MaxPool2d(2, 2)
        )