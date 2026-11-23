import ast
import importlib
import importlib.util
from pathlib import Path
import sys

import pytest
import torch
import torch.nn as nn

basics_path = Path(__file__).parents[1] / "Models" / "Basics.py"
basics_spec = importlib.util.spec_from_file_location("Models.Basics", basics_path)
assert basics_spec is not None
assert basics_spec.loader is not None
basics = importlib.util.module_from_spec(basics_spec)
sys.modules[basics_spec.name] = basics
basics_spec.loader.exec_module(basics)

from Models.Basics import Conv1DBn, ReLUConv2D  # noqa: E402


ND_OPTIONS = {
    "1D": (nn.Conv1d, (2, 8, 11)),
    "2D": (nn.Conv2d, (2, 8, 11, 11)),
}
NORM_OPTIONS = {
    "Bn": {"1D": nn.BatchNorm1d, "2D": nn.BatchNorm2d},
    "In": {"1D": nn.InstanceNorm1d, "2D": nn.InstanceNorm2d},
    "Gn": {"1D": nn.GroupNorm, "2D": nn.GroupNorm},
}
ACT_OPTIONS = {
    "ReLU": nn.ReLU,
    "LeakyReLU": nn.LeakyReLU,
    "GELU": nn.GELU,
    "SiLU": nn.SiLU,
}


def _expectedBlockNames() -> set[str]:
    expected_names = set()
    for nd_name in ND_OPTIONS:
        for norm_name in NORM_OPTIONS:
            expected_names.add(f"Conv{nd_name}{norm_name}")
            expected_names.add(f"{norm_name}Conv{nd_name}")
            for act_name in ACT_OPTIONS:
                expected_names.add(f"Conv{nd_name}{norm_name}{act_name}")
                expected_names.add(f"{norm_name}{act_name}Conv{nd_name}")
        for act_name in ACT_OPTIONS:
            expected_names.add(f"Conv{nd_name}{act_name}")
            expected_names.add(f"{act_name}Conv{nd_name}")
    return expected_names


def _assertBlock(
    class_name: str,
    expected_types: tuple[type[nn.Module], ...],
    input_shape: tuple[int, ...],
    expected_bias: bool,
) -> None:
    block_type = getattr(basics, class_name)
    block = block_type(8, 16, 3, p=1)

    assert block_type.__name__ == class_name
    assert block_type.__module__ == "Models.Basics"
    assert tuple(type(module) for module in block) == expected_types

    conv = next(
        module for module in block if isinstance(module, nn.modules.conv._ConvNd)
    )
    assert (conv.bias is not None) is expected_bias

    output = block(torch.randn(input_shape))
    assert output.shape == (input_shape[0], 16, *input_shape[2:])


def test_blockNamesAreHardCodedAndImportable():
    expected_names = _expectedBlockNames()
    source_tree = ast.parse(basics_path.read_text())
    assigned_names = {
        target.id
        for node in source_tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }

    assert len(expected_names) == 76
    assert expected_names.issubset(assigned_names)
    assert Conv1DBn is getattr(importlib.import_module("Models.Basics"), "Conv1DBn")
    assert ReLUConv2D is getattr(importlib.import_module("Models.Basics"), "ReLUConv2D")


@pytest.mark.parametrize("nd_name", ND_OPTIONS)
@pytest.mark.parametrize("norm_name", NORM_OPTIONS)
@pytest.mark.parametrize("act_name", ACT_OPTIONS)
def test_allNormActivationBlockOrders(nd_name, norm_name, act_name):
    conv_type, input_shape = ND_OPTIONS[nd_name]
    norm_type = NORM_OPTIONS[norm_name][nd_name]
    act_type = ACT_OPTIONS[act_name]

    _assertBlock(
        f"Conv{nd_name}{norm_name}{act_name}",
        (conv_type, norm_type, act_type),
        input_shape,
        expected_bias=False,
    )
    _assertBlock(
        f"{norm_name}{act_name}Conv{nd_name}",
        (norm_type, act_type, conv_type),
        input_shape,
        expected_bias=True,
    )


@pytest.mark.parametrize("nd_name", ND_OPTIONS)
@pytest.mark.parametrize("norm_name", NORM_OPTIONS)
def test_allNormBlockOrders(nd_name, norm_name):
    conv_type, input_shape = ND_OPTIONS[nd_name]
    norm_type = NORM_OPTIONS[norm_name][nd_name]

    _assertBlock(
        f"Conv{nd_name}{norm_name}",
        (conv_type, norm_type),
        input_shape,
        expected_bias=False,
    )
    _assertBlock(
        f"{norm_name}Conv{nd_name}",
        (norm_type, conv_type),
        input_shape,
        expected_bias=True,
    )


@pytest.mark.parametrize("nd_name", ND_OPTIONS)
@pytest.mark.parametrize("act_name", ACT_OPTIONS)
def test_allActivationBlockOrders(nd_name, act_name):
    conv_type, input_shape = ND_OPTIONS[nd_name]
    act_type = ACT_OPTIONS[act_name]

    _assertBlock(
        f"Conv{nd_name}{act_name}",
        (conv_type, act_type),
        input_shape,
        expected_bias=True,
    )
    _assertBlock(
        f"{act_name}Conv{nd_name}",
        (act_type, conv_type),
        input_shape,
        expected_bias=True,
    )


@pytest.mark.parametrize(
    "class_name",
    ("BnReLUConv1D", "BnConv1D", "Conv1DReLU", "ReLUConv1D"),
)
def test_trailingConvolutionBiasCanBeDisabled(class_name):
    block = getattr(basics, class_name)(8, 16, 3, bias=False)
    conv = next(
        module for module in block if isinstance(module, nn.modules.conv._ConvNd)
    )

    assert conv.bias is None


def test_activationFirstBlockDoesNotMutateItsInput():
    block = basics.ReLUConv1D(8, 16, 3, p=1)
    input_tensor = torch.randn(2, 8, 11, requires_grad=True)
    original_input = input_tensor.detach().clone()

    block(input_tensor).sum().backward()

    torch.testing.assert_close(input_tensor.detach(), original_input)
