import torch
import torch.nn as nn


class SELayer2D(nn.Sequential):
    def __init__(self, c_in: int, reduction: int=16):
        """
        Squeeze-and-Excitation Layer, channel wise attention in convolutional neural networks.
        :param c_in: Number of input channels
        :param reduction: Reduction ratio
        """
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_in, c_in // reduction, 1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(c_in // reduction, c_in, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * super().forward(x)


class SELayer1D(nn.Sequential):
    def __init__(self, c_in: int, reduction: int=16):
        """
        Squeeze-and-Excitation Layer, channel wise attention in convolutional neural networks.
        :param c_in: Number of input channels
        :param reduction: Reduction ratio
        """
        super().__init__(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(c_in, c_in // reduction, 1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(c_in // reduction, c_in, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * super().forward(x)


class MHSA(nn.Module):
    def __init__(self, d_in: int, num_heads: int, dropout: float=0.0):
        """
        Multi-head self-attention layer. It includes query, key, value projections and output projection.
        :param d_in: Input dimension.
        :param num_heads: Number of attention heads.
        """
        super(MHSA, self).__init__()

        self.attn = nn.MultiheadAttention(d_in, num_heads, dropout, batch_first=True)

    def forward(self, x):
        return self.attn(x, x, x)[0]