import torch.nn as _nn
import torch as _torch

_norm_options = {
    "Bn": {"1D": _nn.BatchNorm1d, "2D": _nn.BatchNorm2d},
    "In": {"1D": _nn.InstanceNorm1d, "2D": _nn.InstanceNorm2d},
    "Gn": {"1D": _nn.GroupNorm, "2D": _nn.GroupNorm},
}

_activations = {
    "ReLU": _nn.ReLU,
    "LeakyReLU": _nn.LeakyReLU,
    "GELU": _nn.GELU,
    "Swish": _nn.SiLU,
}

def _ConvNDNormActCreator(class_name: str) -> type:
    ND, norm_name, act_name = class_name[4:6], class_name[6:8], class_name[8:]

    conv = _nn.Conv1d if ND == "1D" else _nn.Conv2d
    norm = _norm_options[norm_name][ND]
    act = _activations[act_name]

    def initFunc(self, c_in: int, c_out: int, k: int, s: int=1, p: int=0, d: int=1, g: int=1):
        _nn.Sequential.__init__(self,
                                conv(c_in, c_out, k, s, p, d, g, bias=False),
                                norm(32, c_out) if norm_name == "Gn" else norm(c_out),
                                act(inplace=True)
                                )

    return type(class_name, (_nn.Sequential,), {"__init__": initFunc, "__name__": class_name})

# Conv1D*
# Conv1DBn*
Conv1DBnReLU =      _ConvNDNormActCreator("Conv1DBnReLU")
Conv1DBnLeakyReLU = _ConvNDNormActCreator("Conv1DBnLeakyReLU")
Conv1DBnGELU =      _ConvNDNormActCreator("Conv1DBnGELU")
Conv1DBnSwish =     _ConvNDNormActCreator("Conv1DBnSwish")
# Conv1DIn*
Conv1DInReLU =      _ConvNDNormActCreator("Conv1DInReLU")
Conv1DInLeakyReLU = _ConvNDNormActCreator("Conv1DInLeakyReLU")
Conv1DInGELU =      _ConvNDNormActCreator("Conv1DInGELU")
Conv1DInSwish =     _ConvNDNormActCreator("Conv1DInSwish")
# Conv1DGn*
Conv1DGnReLU =      _ConvNDNormActCreator("Conv1DGnReLU")
Conv1DGnLeakyReLU = _ConvNDNormActCreator("Conv1DGnLeakyReLU")
Conv1DGnGELU =      _ConvNDNormActCreator("Conv1DGnGELU")
Conv1DGnSwish =     _ConvNDNormActCreator("Conv1DGnSwish")

# Conv2D*
# Conv2DBn*
Conv2DBnReLU =      _ConvNDNormActCreator("Conv2DBnReLU")
Conv2DBnLeakyReLU = _ConvNDNormActCreator("Conv2DBnLeakyReLU")
Conv2DBnGELU =      _ConvNDNormActCreator("Conv2DBnGELU")
Conv2DBnSwish =     _ConvNDNormActCreator("Conv2DBnSwish")
# Conv2DIn*
Conv2DInReLU =      _ConvNDNormActCreator("Conv2DInReLU")
Conv2DInLeakyReLU = _ConvNDNormActCreator("Conv2DInLeakyReLU")
Conv2DInGELU =      _ConvNDNormActCreator("Conv2DInGELU")
Conv2DInSwish =     _ConvNDNormActCreator("Conv2DInSwish")
# Conv2DGn*
Conv2DGnReLU =      _ConvNDNormActCreator("Conv2DGnReLU")
Conv2DGnLeakyReLU = _ConvNDNormActCreator("Conv2DGnLeakyReLU")
Conv2DGnGELU =      _ConvNDNormActCreator("Conv2DGnGELU")
Conv2DGnSwish =     _ConvNDNormActCreator("Conv2DGnSwish")


class FCLayers(_nn.Sequential):
    def __init__(self, channel_list: list[int], act: _nn.Module, final_act: _nn.Module=None):
        super().__init__()
        for i in range(len(channel_list) - 1):
            self.append(_nn.Linear(channel_list[i], channel_list[i + 1]))
            if i < len(channel_list) - 2:
                self.append(act)
        if final_act is not None:
            self.append(final_act)


class PosEncoderSinusoidal(_nn.Module):
    def __init__(self, dim: int, max_len: int=5000):
        super().__init__()
        self.dim = dim
        self.max_len = max_len

        self.register_buffer("pe", _torch.zeros(max_len, dim))
        position = _torch.arange(0, max_len, dtype=_torch.float).unsqueeze(1)
        div_term = _torch.exp(_torch.arange(0, dim, 2).float() * (-_torch.log(_torch.tensor(10000.0)) / dim))
        self.pe[:, 0::2] = _torch.sin(position * div_term)
        self.pe[:, 1::2] = _torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)

    def forward(self, x: _torch.Tensor) -> _torch.Tensor:
        """
        Applies sinusoidal positional encoding to the input tensor.
        :param x: Input tensor of shape (batch_size, seq_len, dim).
        :return: Tensor with positional encoding added, of the same shape as input.
        """
        x = x + self.pe[:, :x.size(1)]
        return x


class PosEncoderLearned(_nn.Module):
    def __init__(self, dim: int, max_len: int=5000):
        super().__init__()
        self.dim = dim
        self.max_len = max_len

        self.pe = _nn.Embedding(max_len, dim)
        self.register_buffer("pe_idx", _torch.arange(max_len).unsqueeze(0))

    def forward(self, x: _torch.Tensor) -> _torch.Tensor:
        """
        Applies learned positional encoding to the input tensor.
        :param x: Input tensor of shape (batch_size, seq_len, dim).
        :return: Tensor with positional encoding added, of the same shape as input.
        """
        x = x + self.pe(self.pe_idx[:, :x.size(1)])
        return x


class PosEncoderRotary(_nn.Module):
    """ https://arxiv.org/abs/2104.09864 """
    def __init__(self, dim: int, max_len: int=5000, base: float=10000.0):
        super().__init__()
        self.dim = dim
        self.max_len = max_len

        inv_freq = 1.0 / (base ** (_torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # build cache
        t = _torch.arange(max_len).unsqueeze(1).float()
        freqs = _torch.einsum("i,j->ij", t, inv_freq)
        emb = _torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, :, None, :])
        self.register_buffer("sin_cached", emb.sin()[None, :, None, :])

    def forward(self, x: _torch.Tensor) -> _torch.Tensor:
        seq_len = x.size(1)
        cos = self.cos_cached[:, :seq_len, :, :]
        sin = self.sin_cached[:, :seq_len, :, :]
        x1, x2 = x[..., ::2], x[..., 1::2]
        x = _torch.stack((x1, x2), dim=-1)
        x_rotated = _torch.stack((
            x[..., 0] * cos - x[..., 1] * sin,
            x[..., 1] * cos + x[..., 0] * sin
        ), dim=-1)
        return x_rotated.flatten(-2)
