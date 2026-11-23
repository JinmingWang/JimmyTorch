import torch.nn as _nn
import torch as _torch
from typing import Literal as _Literal

_norm_options = {
    "Bn": {"1D": _nn.BatchNorm1d, "2D": _nn.BatchNorm2d},
    "In": {"1D": _nn.InstanceNorm1d, "2D": _nn.InstanceNorm2d},
    "Gn": {"1D": _nn.GroupNorm, "2D": _nn.GroupNorm},
}

_conv_options = {
    "1D": _nn.Conv1d,
    "2D": _nn.Conv2d,
}

_activations = {
    "ReLU": _nn.ReLU,
    "LeakyReLU": _nn.LeakyReLU,
    "GELU": _nn.GELU,
    "SiLU": _nn.SiLU,
}


def _makeNorm(norm_name: str, nd_name: str, channels: int, gn_groups: int):
    norm = _norm_options[norm_name][nd_name]
    return norm(gn_groups, channels) if norm_name == "Gn" else norm(channels)


def _makeActivation(act_name: str, inplace: bool = True):
    act = _activations[act_name]
    return act() if act_name == "GELU" else act(inplace=inplace)


def _createSequentialClass(class_name: str, init_func) -> type:
    return type(
        class_name,
        (_nn.Sequential,),
        {
            "__init__": init_func,
            "__module__": __name__,
        },
    )


def _convNDNormActCreator(
    class_name: str, nd_name: str, norm_name: str, act_name: str
) -> type:
    conv = _conv_options[nd_name]

    def initFunc(
        self,
        c_in: int,
        c_out: int,
        k: int,
        s: int = 1,
        p: int = 0,
        d: int = 1,
        g: int = 1,
        gn_groups: int = 8,
    ):
        _nn.Sequential.__init__(
            self,
            conv(c_in, c_out, k, s, p, d, g, bias=False),
            _makeNorm(norm_name, nd_name, c_out, gn_groups),
            _makeActivation(act_name),
        )

    return _createSequentialClass(class_name, initFunc)


def _normActConvNDCreator(
    class_name: str, nd_name: str, norm_name: str, act_name: str
) -> type:
    conv = _conv_options[nd_name]

    def initFunc(
        self,
        c_in: int,
        c_out: int,
        k: int,
        s: int = 1,
        p: int = 0,
        d: int = 1,
        g: int = 1,
        gn_groups: int = 8,
        bias: bool = True,
    ):
        _nn.Sequential.__init__(
            self,
            _makeNorm(norm_name, nd_name, c_in, gn_groups),
            _makeActivation(act_name),
            conv(c_in, c_out, k, s, p, d, g, bias=bias),
        )

    return _createSequentialClass(class_name, initFunc)


def _convNormCreator(class_name: str, nd_name: str, norm_name: str) -> type:
    conv = _conv_options[nd_name]

    def initFunc(
        self,
        c_in: int,
        c_out: int,
        k: int,
        s: int = 1,
        p: int = 0,
        d: int = 1,
        g: int = 1,
        gn_groups: int = 8,
    ):
        _nn.Sequential.__init__(
            self,
            conv(c_in, c_out, k, s, p, d, g, bias=False),
            _makeNorm(norm_name, nd_name, c_out, gn_groups),
        )

    return _createSequentialClass(class_name, initFunc)


def _normConvCreator(class_name: str, nd_name: str, norm_name: str) -> type:
    conv = _conv_options[nd_name]

    def initFunc(
        self,
        c_in: int,
        c_out: int,
        k: int,
        s: int = 1,
        p: int = 0,
        d: int = 1,
        g: int = 1,
        gn_groups: int = 8,
        bias: bool = True,
    ):
        _nn.Sequential.__init__(
            self,
            _makeNorm(norm_name, nd_name, c_in, gn_groups),
            conv(c_in, c_out, k, s, p, d, g, bias=bias),
        )

    return _createSequentialClass(class_name, initFunc)


def _convActCreator(class_name: str, nd_name: str, act_name: str) -> type:
    conv = _conv_options[nd_name]

    def initFunc(
        self,
        c_in: int,
        c_out: int,
        k: int,
        s: int = 1,
        p: int = 0,
        d: int = 1,
        g: int = 1,
        bias: bool = True,
    ):
        _nn.Sequential.__init__(
            self,
            conv(c_in, c_out, k, s, p, d, g, bias=bias),
            _makeActivation(act_name),
        )

    return _createSequentialClass(class_name, initFunc)


def _actConvCreator(class_name: str, nd_name: str, act_name: str) -> type:
    conv = _conv_options[nd_name]

    def initFunc(
        self,
        c_in: int,
        c_out: int,
        k: int,
        s: int = 1,
        p: int = 0,
        d: int = 1,
        g: int = 1,
        bias: bool = True,
    ):
        _nn.Sequential.__init__(
            self,
            _makeActivation(act_name, inplace=False),
            conv(c_in, c_out, k, s, p, d, g, bias=bias),
        )

    return _createSequentialClass(class_name, initFunc)


# Conv -> Norm -> Activation
Conv1DBnReLU = _convNDNormActCreator("Conv1DBnReLU", "1D", "Bn", "ReLU")
Conv1DBnLeakyReLU = _convNDNormActCreator("Conv1DBnLeakyReLU", "1D", "Bn", "LeakyReLU")
Conv1DBnGELU = _convNDNormActCreator("Conv1DBnGELU", "1D", "Bn", "GELU")
Conv1DBnSiLU = _convNDNormActCreator("Conv1DBnSiLU", "1D", "Bn", "SiLU")
Conv1DInReLU = _convNDNormActCreator("Conv1DInReLU", "1D", "In", "ReLU")
Conv1DInLeakyReLU = _convNDNormActCreator("Conv1DInLeakyReLU", "1D", "In", "LeakyReLU")
Conv1DInGELU = _convNDNormActCreator("Conv1DInGELU", "1D", "In", "GELU")
Conv1DInSiLU = _convNDNormActCreator("Conv1DInSiLU", "1D", "In", "SiLU")
Conv1DGnReLU = _convNDNormActCreator("Conv1DGnReLU", "1D", "Gn", "ReLU")
Conv1DGnLeakyReLU = _convNDNormActCreator("Conv1DGnLeakyReLU", "1D", "Gn", "LeakyReLU")
Conv1DGnGELU = _convNDNormActCreator("Conv1DGnGELU", "1D", "Gn", "GELU")
Conv1DGnSiLU = _convNDNormActCreator("Conv1DGnSiLU", "1D", "Gn", "SiLU")
Conv2DBnReLU = _convNDNormActCreator("Conv2DBnReLU", "2D", "Bn", "ReLU")
Conv2DBnLeakyReLU = _convNDNormActCreator("Conv2DBnLeakyReLU", "2D", "Bn", "LeakyReLU")
Conv2DBnGELU = _convNDNormActCreator("Conv2DBnGELU", "2D", "Bn", "GELU")
Conv2DBnSiLU = _convNDNormActCreator("Conv2DBnSiLU", "2D", "Bn", "SiLU")
Conv2DInReLU = _convNDNormActCreator("Conv2DInReLU", "2D", "In", "ReLU")
Conv2DInLeakyReLU = _convNDNormActCreator("Conv2DInLeakyReLU", "2D", "In", "LeakyReLU")
Conv2DInGELU = _convNDNormActCreator("Conv2DInGELU", "2D", "In", "GELU")
Conv2DInSiLU = _convNDNormActCreator("Conv2DInSiLU", "2D", "In", "SiLU")
Conv2DGnReLU = _convNDNormActCreator("Conv2DGnReLU", "2D", "Gn", "ReLU")
Conv2DGnLeakyReLU = _convNDNormActCreator("Conv2DGnLeakyReLU", "2D", "Gn", "LeakyReLU")
Conv2DGnGELU = _convNDNormActCreator("Conv2DGnGELU", "2D", "Gn", "GELU")
Conv2DGnSiLU = _convNDNormActCreator("Conv2DGnSiLU", "2D", "Gn", "SiLU")

# Norm -> Activation -> Conv
BnReLUConv1D = _normActConvNDCreator("BnReLUConv1D", "1D", "Bn", "ReLU")
BnLeakyReLUConv1D = _normActConvNDCreator("BnLeakyReLUConv1D", "1D", "Bn", "LeakyReLU")
BnGELUConv1D = _normActConvNDCreator("BnGELUConv1D", "1D", "Bn", "GELU")
BnSiLUConv1D = _normActConvNDCreator("BnSiLUConv1D", "1D", "Bn", "SiLU")
InReLUConv1D = _normActConvNDCreator("InReLUConv1D", "1D", "In", "ReLU")
InLeakyReLUConv1D = _normActConvNDCreator("InLeakyReLUConv1D", "1D", "In", "LeakyReLU")
InGELUConv1D = _normActConvNDCreator("InGELUConv1D", "1D", "In", "GELU")
InSiLUConv1D = _normActConvNDCreator("InSiLUConv1D", "1D", "In", "SiLU")
GnReLUConv1D = _normActConvNDCreator("GnReLUConv1D", "1D", "Gn", "ReLU")
GnLeakyReLUConv1D = _normActConvNDCreator("GnLeakyReLUConv1D", "1D", "Gn", "LeakyReLU")
GnGELUConv1D = _normActConvNDCreator("GnGELUConv1D", "1D", "Gn", "GELU")
GnSiLUConv1D = _normActConvNDCreator("GnSiLUConv1D", "1D", "Gn", "SiLU")
BnReLUConv2D = _normActConvNDCreator("BnReLUConv2D", "2D", "Bn", "ReLU")
BnLeakyReLUConv2D = _normActConvNDCreator("BnLeakyReLUConv2D", "2D", "Bn", "LeakyReLU")
BnGELUConv2D = _normActConvNDCreator("BnGELUConv2D", "2D", "Bn", "GELU")
BnSiLUConv2D = _normActConvNDCreator("BnSiLUConv2D", "2D", "Bn", "SiLU")
InReLUConv2D = _normActConvNDCreator("InReLUConv2D", "2D", "In", "ReLU")
InLeakyReLUConv2D = _normActConvNDCreator("InLeakyReLUConv2D", "2D", "In", "LeakyReLU")
InGELUConv2D = _normActConvNDCreator("InGELUConv2D", "2D", "In", "GELU")
InSiLUConv2D = _normActConvNDCreator("InSiLUConv2D", "2D", "In", "SiLU")
GnReLUConv2D = _normActConvNDCreator("GnReLUConv2D", "2D", "Gn", "ReLU")
GnLeakyReLUConv2D = _normActConvNDCreator("GnLeakyReLUConv2D", "2D", "Gn", "LeakyReLU")
GnGELUConv2D = _normActConvNDCreator("GnGELUConv2D", "2D", "Gn", "GELU")
GnSiLUConv2D = _normActConvNDCreator("GnSiLUConv2D", "2D", "Gn", "SiLU")

# Conv -> Norm
Conv1DBn = _convNormCreator("Conv1DBn", "1D", "Bn")
Conv1DIn = _convNormCreator("Conv1DIn", "1D", "In")
Conv1DGn = _convNormCreator("Conv1DGn", "1D", "Gn")
Conv2DBn = _convNormCreator("Conv2DBn", "2D", "Bn")
Conv2DIn = _convNormCreator("Conv2DIn", "2D", "In")
Conv2DGn = _convNormCreator("Conv2DGn", "2D", "Gn")

# Norm -> Conv
BnConv1D = _normConvCreator("BnConv1D", "1D", "Bn")
InConv1D = _normConvCreator("InConv1D", "1D", "In")
GnConv1D = _normConvCreator("GnConv1D", "1D", "Gn")
BnConv2D = _normConvCreator("BnConv2D", "2D", "Bn")
InConv2D = _normConvCreator("InConv2D", "2D", "In")
GnConv2D = _normConvCreator("GnConv2D", "2D", "Gn")

# Conv -> Activation
Conv1DReLU = _convActCreator("Conv1DReLU", "1D", "ReLU")
Conv1DLeakyReLU = _convActCreator("Conv1DLeakyReLU", "1D", "LeakyReLU")
Conv1DGELU = _convActCreator("Conv1DGELU", "1D", "GELU")
Conv1DSiLU = _convActCreator("Conv1DSiLU", "1D", "SiLU")
Conv2DReLU = _convActCreator("Conv2DReLU", "2D", "ReLU")
Conv2DLeakyReLU = _convActCreator("Conv2DLeakyReLU", "2D", "LeakyReLU")
Conv2DGELU = _convActCreator("Conv2DGELU", "2D", "GELU")
Conv2DSiLU = _convActCreator("Conv2DSiLU", "2D", "SiLU")

# Activation -> Conv
ReLUConv1D = _actConvCreator("ReLUConv1D", "1D", "ReLU")
LeakyReLUConv1D = _actConvCreator("LeakyReLUConv1D", "1D", "LeakyReLU")
GELUConv1D = _actConvCreator("GELUConv1D", "1D", "GELU")
SiLUConv1D = _actConvCreator("SiLUConv1D", "1D", "SiLU")
ReLUConv2D = _actConvCreator("ReLUConv2D", "2D", "ReLU")
LeakyReLUConv2D = _actConvCreator("LeakyReLUConv2D", "2D", "LeakyReLU")
GELUConv2D = _actConvCreator("GELUConv2D", "2D", "GELU")
SiLUConv2D = _actConvCreator("SiLUConv2D", "2D", "SiLU")


class FCLayers(_nn.Sequential):
    def __init__(
        self, channel_list: list[int], act: _nn.Module, final_act: _nn.Module = None
    ):
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
    def __init__(
        self,
        dim: int,
        max_len: int = 5000,
        merge_mode: _Literal["add", "concat"] = "add",
        d_pe: int = None,
    ):
        """

        :param dim: The dimension of the sequence to be encoded.
        :param max_len: Maximum length of the sequence for which positional encoding is computed.
        :param merge_mode: _Literal["add", "concat"]
        :param d_pe: When use concat, the dimension of positional encoding.
        """
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.merge_mode = merge_mode

        if merge_mode == "add":
            d_pe = dim
        self.d_pe = dim if d_pe is None else d_pe

        self.register_buffer("pe", _torch.zeros(max_len, self.d_pe))
        position = _torch.arange(0, max_len, dtype=_torch.float).unsqueeze(1)
        div_term = _torch.exp(
            _torch.arange(0, self.d_pe, 2).float()
            * (-_torch.log(_torch.tensor(10000.0)) / self.d_pe)
        )
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
    def __init__(
        self,
        dim: int,
        max_len: int = 5000,
        merge_mode: _Literal["add", "concat"] = "add",
    ):
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
            pe_idx = self.pe(_torch.arange(x.shape[1], device=x.device)).unsqueeze(0)
            return x + pe_idx
        else:
            B = x.size(0)
            return _torch.cat(
                (x, self.pe(self.pe_idx[:, : x.size(1)]).repeat(B, 1, 1)), dim=-1
            )


class PosEncoderRotary(_nn.Module):
    """https://arxiv.org/abs/2104.09864"""

    def __init__(self, dim: int, max_len: int = 5000, base: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"PosEncoderRotary requires an even dim, got {dim}.")
        self.dim = dim
        self.max_len = max_len

        inv_freq = 1.0 / (base ** (_torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # build cache
        t = _torch.arange(max_len).float()
        freqs = _torch.einsum("i,j->ij", t, inv_freq)
        self.register_buffer("cos_cached", freqs.cos()[None, :, :])
        self.register_buffer("sin_cached", freqs.sin()[None, :, :])

    def forward(self, x: _torch.Tensor) -> _torch.Tensor:
        seq_len = x.size(1)
        cos = self.cos_cached[:, :seq_len, :].to(dtype=x.dtype, device=x.device)
        sin = self.sin_cached[:, :seq_len, :].to(dtype=x.dtype, device=x.device)
        while cos.ndim < x.ndim:
            cos = cos.unsqueeze(-2)
            sin = sin.unsqueeze(-2)
        x1, x2 = x[..., ::2], x[..., 1::2]
        x = _torch.stack((x1, x2), dim=-1)
        x_rotated = _torch.stack(
            (x[..., 0] * cos - x[..., 1] * sin, x[..., 1] * cos + x[..., 0] * sin),
            dim=-1,
        )
        return x_rotated.flatten(-2)


class PatchMaker1D(_nn.Module):
    """
    Patch maker for 1D data.
    :param patch_size: Size of each patch.
    :param stride: Stride for the sliding window.
    :param patch_as_vector: If True, patches are flattened into vectors.
    """

    def __init__(self, patch_size: int, stride: int = 1, patch_as_vector: bool = True):
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
        patches = (
            x.unfold(1, self.patch_size, self.stride).transpose(-1, -2).contiguous()
        )
        # Now patches: (B, num_patches, patch_size, dim)
        if self.patch_as_vector:
            return patches.flatten(2)  # (B, num_patches, patch_size * dim)
        return patches


class PatchMaker2D(_nn.Module):
    """
    Patch maker for 2D data.
    :param patch_size: Size of each patch.
    :param stride: Stride for the sliding window.
    :param patch_as_vector: If True, patches are flattened into vectors.
    """

    def __init__(
        self,
        patch_size: int,
        stride: int = 1,
        patch_as_vector: bool = True,
        flatten: bool = True,
    ):
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
        patches = (
            x.unfold(2, self.patch_size, self.stride)
            .unfold(3, self.patch_size, self.stride)
            .contiguous()
        )
        # Now patches: (B, C, num_patches_h, num_patches_w, patch_size, patch_size)
        patches = patches.permute(0, 4, 5, 1, 2, 3).contiguous()
        # Now patches: (B, patch_size, patch_size, C, num_patches_h, num_patches_w)
        if self.patch_as_vector:
            patches = patches.flatten(
                1, 3
            )  # (B, patch_size * patch_size * C, num_patches_h, num_patches_w)
        if self.flatten:
            patches = patches.flatten(
                2
            )  # (B, patch_size * patch_size * C, num_patches_h * num_patches_w)
        return patches


if __name__ == "__main__":
    x = _torch.randn(1, 64, 4)
    y = PatchMaker1D(8, 8, False)(x)
    print(y.shape)
