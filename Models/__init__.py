from .Basics import *
from .CostTest import testCudaSpeed, testFlops
from .Attentions import SELayer1D, SELayer2D
from .ModelUtils import Transpose, Permute, Reshape, PrintShape, SequentialMultiIO, makeItResidual
from .JimmyModel import JimmyModel

from .SampleCNN import SampleCNN