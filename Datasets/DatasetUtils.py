import torch
from typing import Any, Literal
import torchvision
from rich import print as rprint

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")