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
            if not isinstance(dynamic_inputs, tuple):
                dynamic_inputs = (dynamic_inputs,)
        return dynamic_inputs


class Rearrange(nn.Module):
    """
    A module that rearranges the input tensor according to the specified pattern.
    :param pattern: The pattern to rearrange the input tensor.
    """
    def __init__(self, pattern: str, **kwargs):
        super(Rearrange, self).__init__()
        src_pattern, dst_pattern = pattern.split(" -> ")

        operations = []

        # --- STEP 1. locate all the brackets in the source pattern
        brackets = []
        dim_idx = 0
        in_bracket = False
        for i, c in enumerate(src_pattern):
            if c == "(":
                brackets.append([i + 1, None, dim_idx])
                in_bracket = True
            elif c == ")":
                brackets[-1][1] = i
                in_bracket = False
            elif c == " " and not in_bracket:
                dim_idx += 1

        # --- STEP 2. break (unflatten) all brackets from back to front
        for i in range(len(brackets) - 1, -1, -1):
            start, end, dim_idx = brackets[i]

            dims_within = src_pattern[start:end].split(" ")     # how many dimensions within the brackets

            # need at least dims_within - 1 known dimensions
            known_dims = [-1] * len(dims_within)
            for dim_i, dim_symbol in enumerate(dims_within):
                if dim_symbol in kwargs:
                    known_dims[dim_i] = kwargs[dim_symbol]
            if known_dims.count(-1) > 1:
                raise ValueError(f"Cannot unflatten {src_pattern[start:end]} because too many unknown dimensions.")

            operations.append(nn.Unflatten(dim_idx, tuple(known_dims)))

        # --- STEP 3. Permute the dimensions in src to match the dst pattern
        src_symbols = src_pattern.replace("(", "").replace(")", "").split(" ")
        dst_symbols = dst_pattern.replace("(", "").replace(")", "").split(" ")
        permute_dims = []
        for dim_symbol in dst_symbols:
            if dim_symbol in src_symbols:
                permute_dims.append(src_symbols.index(dim_symbol))
            else:
                raise ValueError(f"Cannot permute {dim_symbol} because it is not in the source pattern.")
        operations.append(Permute(*permute_dims))

        # --- STEP 4. locate all the brackets in the destination pattern
        brackets = []
        dim_idx = 0
        for i, c in enumerate(dst_pattern):
            if c == "(":
                brackets.append([dim_idx, None])
            elif c == ")":
                brackets[-1][1] = dim_idx
            elif c == " ":
                dim_idx += 1

        # --- STEP 5. Merge (flatten) all brackets from back to front
        for i in range(len(brackets) - 1, -1, -1):
            start, end = brackets[i]
            operations.append(nn.Flatten(start, end))

        # --- STEP 6. Create the final operation
        self.operations = nn.Sequential(*operations)


    def forward(self, x):
        """
        Forward pass through the rearrange module.
        :param x: The input tensor.
        :return: The rearranged tensor.
        """
        return self.operations(x)


def makeItResidual(forward_func):
    """
    Make a forward function residual.
    :param forward_func: The function to make residual.
    :return: The residual function.
    """
    def residual_func(self, x):
        return x + forward_func(self, x)

    return residual_func