from .MovingAverage import MovingAvg, MovingAvgGroup
from .ProgressManagerGUI import ProgressManagerGUI

try:
	from .ProgressManager import ProgressManager
except ModuleNotFoundError as error:
	if error.name != "rich":
		raise

try:
	from .TensorBoardManager import TensorBoardManager
except ModuleNotFoundError as error:
	if error.name != "tensorboard":
		raise