import torch

def extendAs(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Extend tensor a to the same number of dimensions as tensor b by adding singleton dimensions.
    :param a: Tensor to be extended.
    :param b: Tensor to match shape with.
    :return: Extended tensor a.
    """
    if a.ndim < b.ndim:
        return a.view(a.shape + (1,) * (b.ndim - a.ndim))