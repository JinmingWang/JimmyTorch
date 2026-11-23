from .JimmyDataset import JimmyDataset
from .MNISTDataset import MNISTSampleDataset
from .MultiThreadLoader import MultiThreadLoader
from .DatasetUtils import DEVICE

from .TrajectoryUtils import Traj, BatchTraj, getLat, getLng, computeDistance, cropPadTraj, flipTrajWestEast, \
    flipTrajNorthSouth, centerTraj, zScoreTraj, minMaxTraj, rotateTraj, interpTraj, plotTraj, geometricDistance, computeJSD

from .SequenceUtils import cropPadSequence

from importlib import import_module as _import_module
from ._mesh_exports import MESH_MODULES as _mesh_modules

from typing import *
import torch


# Keep the existing "from Datasets import *" API, including the mesh helpers.
__all__ = [name for name in globals() if not name.startswith("_")] + ["ThreeDUtils", *_mesh_modules]


def __getattr__(name):
    if name == "ThreeDUtils" or name in _mesh_modules:
        mesh_utils = _import_module(".ThreeDUtils", __name__)
        return mesh_utils if name == "ThreeDUtils" else getattr(mesh_utils, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
