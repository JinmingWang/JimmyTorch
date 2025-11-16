import torch
from contextlib import nullcontext

def extendAs(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Extend tensor a to the same number of dimensions as tensor b by adding singleton dimensions.
    :param a: Tensor to be extended.
    :param b: Tensor to match shape with.
    :return: Extended tensor a.
    """
    if a.ndim < b.ndim:
        return a.view(a.shape + (1,) * (b.ndim - a.ndim))


def getAutoCast(data_sample: torch.Tensor, mixed_precision: bool):
    if mixed_precision:
        return torch.autocast(device_type=data_sample.device, dtype=torch.float16)
    else:
        return nullcontext()