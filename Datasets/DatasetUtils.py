import torch
from typing import *
from jaxtyping import Float as FT32
import torchvision
from rich import print as rprint

Tensor = torch.Tensor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")