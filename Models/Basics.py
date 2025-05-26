import torch.nn as _nn
import torch as _torch
from typing import Literal as _Literal

_norm_options = {
    "Bn": {"1D": _nn.BatchNorm1d, "2D": _nn.BatchNorm2d},
    "In": {"1D": _nn.InstanceNorm1d, "2D": _nn.InstanceNorm2d},
    "Gn": {"1D": _nn.GroupNorm, "2D": _nn.GroupNorm},
}

_activations = {
    "ReLU": _nn.ReLU,
    "LeakyReLU": _nn.LeakyReLU,
    "GELU": _nn.GELU,
    "SiLU": _nn.SiLU,
}

def _ConvNDNormActCreator(class_name: str) -> type:
    ND, norm_name, act_name = class_name[4:6], class_name[6:8], class_name[8:]

    conv = _nn.Conv1d if ND == "1D" else _nn.Conv2d
    norm = _norm_options[norm_name][ND]
    act = _activations[act_name]

    def initFunc(self, c_in: int, c_out: int, k: int, s: int=1, p: int=0, d: int=1, g: int=1, gn_groups: int=8):
        _nn.Sequential.__init__(self,
                                conv(c_in, c_out, k, s, p, d, g, bias=False),
                                norm(gn_groups, c_out) if norm_name == "Gn" else norm(c_out),
                                act(inplace=True)
                                )

    return type(class_name, (_nn.Sequential,), {"__init__": initFunc, "__name__": class_name})

# Conv1D*
# Conv1DBn*
Conv1DBnReLU =      _ConvNDNormActCreator("Conv1DBnReLU")
Conv1DBnLeakyReLU = _ConvNDNormActCreator("Conv1DBnLeakyReLU")
Conv1DBnGELU =      _ConvNDNormActCreator("Conv1DBnGELU")
Conv1DBnSiLU =     _ConvNDNormActCreator("Conv1DBnSiLU")
# Conv1DIn*
Conv1DInReLU =      _ConvNDNormActCreator("Conv1DInReLU")
Conv1DInLeakyReLU = _ConvNDNormActCreator("Conv1DInLeakyReLU")
Conv1DInGELU =      _ConvNDNormActCreator("Conv1DInGELU")
Conv1DInSiLU =     _ConvNDNormActCreator("Conv1DInSiLU")
# Conv1DGn*
Conv1DGnReLU =      _ConvNDNormActCreator("Conv1DGnReLU")
Conv1DGnLeakyReLU = _ConvNDNormActCreator("Conv1DGnLeakyReLU")
Conv1DGnGELU =      _ConvNDNormActCreator("Conv1DGnGELU")
Conv1DGnSiLU =     _ConvNDNormActCreator("Conv1DGnSiLU")

# Conv2D*
# Conv2DBn*
Conv2DBnReLU =      _ConvNDNormActCreator("Conv2DBnReLU")
Conv2DBnLeakyReLU = _ConvNDNormActCreator("Conv2DBnLeakyReLU")
Conv2DBnGELU =      _ConvNDNormActCreator("Conv2DBnGELU")
Conv2DBnSiLU =     _ConvNDNormActCreator("Conv2DBnSiLU")
# Conv2DIn*
Conv2DInReLU =      _ConvNDNormActCreator("Conv2DInReLU")
Conv2DInLeakyReLU = _ConvNDNormActCreator("Conv2DInLeakyReLU")
Conv2DInGELU =      _ConvNDNormActCreator("Conv2DInGELU")
Conv2DInSiLU =     _ConvNDNormActCreator("Conv2DInSiLU")
# Conv2DGn*
Conv2DGnReLU =      _ConvNDNormActCreator("Conv2DGnReLU")
Conv2DGnLeakyReLU = _ConvNDNormActCreator("Conv2DGnLeakyReLU")
Conv2DGnGELU =      _ConvNDNormActCreator("Conv2DGnGELU")
Conv2DGnSiLU =     _ConvNDNormActCreator("Conv2DGnSiLU")

def _NormActConvNDCreator(class_name: str) -> type:
    norm_name, act_name, ND = class_name[:2], class_name[2:-6], class_name[-2:]

    conv = _nn.Conv1d if ND == "1D" else _nn.Conv2d
    norm = _norm_options[norm_name][ND]
    act = _activations[act_name]

    def initFunc(self, c_in: int, c_out: int, k: int, s: int=1, p: int=0, d: int=1, g: int=1, gn_groups: int=8):
        _nn.Sequential.__init__(self,
                                norm(gn_groups, c_in) if norm_name == "Gn" else norm(c_in),
                                act() if act_name == "GELU" else act(inplace=True),
                                conv(c_in, c_out, k, s, p, d, g)
                                )

    return type(class_name, (_nn.Sequential,), {"__init__": initFunc, "__name__": class_name})

# *Conv1D
# Bn*Conv1D
BnReLUConv1D =      _NormActConvNDCreator("BnReLUConv1D")
BnLeakyReLUConv1D = _NormActConvNDCreator("BnLeakyReLUConv1D")
BnGELUConv1D =      _NormActConvNDCreator("BnGELUConv1D")
BnSiLUConv1D =     _NormActConvNDCreator("BnSiLUConv1D")
# In*Conv1D
InReLUConv1D =      _NormActConvNDCreator("InReLUConv1D")
InLeakyReLUConv1D = _NormActConvNDCreator("InLeakyReLUConv1D")
InGELUConv1D =      _NormActConvNDCreator("InGELUConv1D")
InSiLUConv1D =     _NormActConvNDCreator("InSiLUConv1D")
# Gn*Conv1D
GnReLUConv1D =      _NormActConvNDCreator("GnReLUConv1D")
GnLeakyReLUConv1D = _NormActConvNDCreator("GnLeakyReLUConv1D")
GnGELUConv1D =      _NormActConvNDCreator("GnGELUConv1D")
GnSiLUConv1D =     _NormActConvNDCreator("GnSiLUConv1D")

# *Conv2D
# Bn*Conv2D
BnReLUConv2D =      _NormActConvNDCreator("BnReLUConv2D")
BnLeakyReLUConv2D = _NormActConvNDCreator("BnLeakyReLUConv2D")
BnGELUConv2D =      _NormActConvNDCreator("BnGELUConv2D")
BnSiLUConv2D =     _NormActConvNDCreator("BnSiLUConv2D")
# In*Conv2D
InReLUConv2D =      _NormActConvNDCreator("InReLUConv2D")
InLeakyReLUConv2D = _NormActConvNDCreator("InLeakyReLUConv2D")
InGELUConv2D =      _NormActConvNDCreator("InGELUConv2D")
InSiLUConv2D =     _NormActConvNDCreator("InSiLUConv2D")
# Gn*Conv2D
GnReLUConv2D =      _NormActConvNDCreator("GnReLUConv2D")
GnLeakyReLUConv2D = _NormActConvNDCreator("GnLeakyReLUConv2D")
GnGELUConv2D =      _NormActConvNDCreator("GnGELUConv2D")
GnSiLUConv2D =     _NormActConvNDCreator("GnSiLUConv2D")



class FCLayers(_nn.Sequential):
    def __init__(self, channel_list: list[int], act: _nn.Module, final_act: _nn.Module=None):
        super().__init__()
        for i in range(len(channel_list) - 1):
            self.append(_nn.Linear(channel_list[i], channel_list[i + 1]))
            if i < len(channel_list) - 2:
                self.append(act)
        if final_act is not None:
            self.append(final_act)


# Just another name for FCLayers
MLP = FCLayers


class PosEncoderSinusoidal(_nn.Module):
    def __init__(self, dim: int, max_len: int=5000, merge_mode: _Literal["add", "concat"] = "add"):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.merge_mode = merge_mode

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
        B, L, D = x.shape
        if self.merge_mode == "add":
            return x + self.pe[:, :L]
        else:
            return _torch.cat((x, self.pe[:, :L].repeat(B, 1, 1)), dim=-1)


class PosEncoderLearned(_nn.Module):
    def __init__(self, dim: int, max_len: int=5000, merge_mode: _Literal["add", "concat"] = "add"):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.merge_mode = merge_mode

        self.pe = _nn.Embedding(max_len, dim)
        self.register_buffer("pe_idx", _torch.arange(max_len).unsqueeze(0))

    def forward(self, x: _torch.Tensor) -> _torch.Tensor:
        """
        Applies learned positional encoding to the input tensor.
        :param x: Input tensor of shape (batch_size, seq_len, dim).
        :return: Tensor with positional encoding added, of the same shape as input.
        """
        if self.merge_mode == "add":
            return x + self.pe(self.pe_idx[:, :x.size(1)])
        else:
            return _torch.cat((x, self.pe(self.pe_idx[:, :x.size(1)])), dim=-1)


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


class PatchMaker1D(_nn.Module):
    """
    Patch maker for 1D data.
    :param patch_size: Size of each patch.
    :param stride: Stride for the sliding window.
    :param patch_as_vector: If True, patches are flattened into vectors.
    """
    def __init__(self, patch_size: int, stride: int=1, patch_as_vector: bool=True):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.patch_as_vector = patch_as_vector

    def forward(self, x: _torch.Tensor) -> _torch.Tensor:
        """
        Applies the patch maker to the input tensor.
        :param x: Input tensor of shape (batch_size, seq_len, dim).
        :return: Tensor with patches extracted.
        """
        patches = x.unfold(1, self.patch_size, self.stride).transpose(-1, -2).contiguous()
        # Now patches: (B, num_patches, patch_size, dim)
        if self.patch_as_vector:
            return patches.flatten(2)    # (B, num_patches, patch_size * dim)
        return patches


class PatchMaker2D(_nn.Module):
    """
    Patch maker for 2D data.
    :param patch_size: Size of each patch.
    :param stride: Stride for the sliding window.
    :param patch_as_vector: If True, patches are flattened into vectors.
    """
    def __init__(self, patch_size: int, stride: int=1, patch_as_vector: bool=True, flatten: bool=True):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.patch_as_vector = patch_as_vector
        self.flatten = flatten

    def forward(self, x: _torch.Tensor) -> _torch.Tensor:
        """
        Applies the patch maker to the input tensor.
        :param x: Input tensor of shape (batch_size, channels, height, width).
        :return: Tensor with patches extracted.
        """
        patches = x.unfold(2, self.patch_size, self.stride).unfold(3, self.patch_size, self.stride).contiguous()
        # Now patches: (B, C, num_patches_h, num_patches_w, patch_size, patch_size)
        patches = patches.permute(0, 4, 5, 1, 2, 3).contiguous()
        # Now patches: (B, patch_size, patch_size, C, num_patches_h, num_patches_w)
        if self.patch_as_vector:
            patches = patches.flatten(1, 3)    # (B, patch_size * patch_size * C, num_patches_h, num_patches_w)
        if self.flatten:
            patches = patches.flatten(2)    # (B, patch_size * patch_size * C, num_patches_h * num_patches_w)
        return patches


if __name__ == '__main__':
    x = _torch.randn(1, 64, 4)
    y = PatchMaker1D(8, 8, False)(x)
    print(y.shape)