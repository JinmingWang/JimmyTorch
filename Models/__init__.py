"""
This module contains the model classes and functions for the project.
"""

from .Basics import *
from .CostTest import testCudaSpeed, testFlops
from .Attentions import SELayer1D, SELayer2D, MHSA
from .ModelUtils import Transpose, Permute, Reshape, PrintShape, SequentialMultiIO, makeItResidual, Rearrange
from .LossFunctions import MaskedLoss, SequentialLossWithLength, RMSE, KLDLoss
from .Functional import extendAs
from .JimmyModel import JimmyModel

from .SampleCNN import SampleCNN

import torch
import torch.nn as nn
import torch.nn.functional as func

from typing import *

Tensor = torch.Tensor