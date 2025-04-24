from .DatasetUtils import *
import matplotlib.pyplot as plt

def cropPadSequence(seq: Tensor, target_len: int, pad_value: float = 0.0) -> Tensor:
    """
    Pad a sequence to the maximum length.

    :param seq: Sequence to be padded, expected to be of shape (L, D) where L is the length and D is the dimension.
    :param target_len: Maximum length of the sequence.
    :param pad_value: Value to pad with. Default is 0.0.
    :return: Padded sequence.
    """
    if seq.shape[0] >= target_len:
        return seq[:target_len]
    elif seq.shape[0] < target_len:
        pad_size = target_len - seq.shape[0]
        return torch.nn.functional.pad(seq, (0, 0, 0, pad_size), value=pad_value)
    else:
        return seq