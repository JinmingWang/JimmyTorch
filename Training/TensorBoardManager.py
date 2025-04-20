from torch.utils.tensorboard import SummaryWriter
from typing import Literal
import os

class TensorBoardManager:
    def __init__(self, log_dir: str, tags: list[str] = None, value_types: list[str] = None):
        """
        Initialize the TensorBoard manager.
        :param log_dir: Directory to save TensorBoard logs.
        """
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.tag_registry = {}
        if tags is not None and value_types is not None:
            if len(tags) != len(value_types):
                raise ValueError("Length of tags and value_types must be the same.")
            for tag, value_type in zip(tags, value_types):
                self.register(tag, value_type)

    def register(self, tag: str, value_type: Literal['scalar', 'figure']):
        """
        Register a new tag for logging.
        :param tag: The tag to register.
        :param value_type: The type of value to log ('scalar' or 'figure').
        """
        if tag in self.tag_registry:
            raise ValueError(f"Tag '{tag}' is already registered.")

        match value_type:
            case 'scalar':
                self.tag_registry[tag] = self.writer.add_scalar
            case 'figure':
                self.tag_registry[tag] = self.writer.add_figure
            case _:
                raise ValueError(f"Unsupported value type '{value_type}'.")

    def log(self, global_step: int, **values):
        """
        Log values to TensorBoard.
        :param values: Keyword arguments where keys are tags and values are the corresponding values to log.
        """
        for tag, value in values.items():
            self.tag_registry[tag](tag, value, global_step=global_step)


