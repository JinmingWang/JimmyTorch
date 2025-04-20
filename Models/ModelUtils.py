import torch
import torch.nn as nn


class Transpose(nn.Module):
    def __init__(self, dim1: int, dim2: int):
        """
        Make transpose function a nn.Module, so it can be used in nn.Sequential.
        :param dim1: The first dimension to transpose.
        :param dim2: The second dimension to transpose.
        """
        super(Transpose, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)


class Permute(nn.Module):
    def __init__(self, *dims: int):
        """
        Make permute function a nn.Module, so it can be used in nn.Sequential.
        :param dims: The dimensions to permute to.
        """
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


class Reshape(nn.Module):
    def __init__(self, *shape: int):
        """
        Make reshape function a nn.Module, so it can be used in nn.Sequential.
        :param shape: The shape to reshape to.
        """
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(*self.shape)


class PrintShape(nn.Module):
    def __init__(self, name: str = ""):
        """
        Print the shape of the input tensor, can be inserted into model for debugging.
        :param name: The name of the tensor to print.
        """
        super(PrintShape, self).__init__()
        self.name = name

    def forward(self, x):
        print(f"{self.name}: {x.shape}")
        return x


class SequentialMultiIO(nn.Sequential):
    """
    A sequential module that can take multiple inputs and outputs.
    :param modules: The modules to apply.
    """
    def forward(self, *dynamic_inputs, **static_inputs):
        """
        Forward pass through the sequential module.
        :param dynamic_inputs: the input values that changes with each forward pass.
        :param static_inputs: the input values that are constant, usually some kind of context.
        :return: The output of the last module.
        """
        for module in self:
            dynamic_inputs = module(*dynamic_inputs, **static_inputs)
        return dynamic_inputs


def makeItResidual(forward_func):
    """
    Make a forward function residual.
    :param forward_func: The function to make residual.
    :return: The residual function.
    """
    def residual_func(self, x):
        return x + forward_func(self, x)

    return residual_func