import torch.nn as nn
import torch
from torch import Tensor
from typing import Callable

class MaskedLoss(nn.Module):
    def __init__(self, base_loss: Callable):
        """
        Initialize the masked loss function.

        :param base_loss: The base loss function to be used.
        """
        super(MaskedLoss, self).__init__()
        self.base_loss = base_loss

    def forward(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        """

        :param pred: the predicted output
        :param target: the target output
        :param mask: the mask to be applied
        :return: the masked loss
        """
        # Apply the mask to the predictions and targets

        mask = (mask > 0).repeat(1, 1, 2)

        if isinstance(target, Tensor):
            masked_pred = pred[mask]
            masked_target = target[mask]
        elif isinstance(target, list):
            masked_pred = pred[mask]
            masked_target = torch.cat(target, dim=0).flatten()
            # Then target is a list of tensors, each item has different shape (L, 2)
            # We can convert both pred and target to sequences of points

        loss = self.base_loss(masked_pred, masked_target)

        return loss


class SequentialLossWithLength(nn.Module):
    def __init__(self, base_loss: nn.Module) -> None:
        """
        Loss applied to sequential data with a length to each data sample.

        :param base_loss: The base loss function to be used.
        """
        super(SequentialLossWithLength, self).__init__()
        self.base_loss = base_loss

    def forward(self, pred: Tensor, target: Tensor, lengths: Tensor) -> Tensor:
        """
        :param pred: the predicted output of shape (B, L, D)
        :param target: the target output of shape (B, L, D)
        :param lengths: the lengths of each data sample of shape (B, )
        :return: the masked loss
        """

        # Create a mask based on the lengths
        mask = torch.arange(pred.size(1), device=pred.device).expand(len(lengths), pred.size(1)) < lengths.unsqueeze(1)

        # Apply the mask to the predictions and targets
        masked_pred = pred * mask.unsqueeze(-1)
        masked_target = target * mask.unsqueeze(-1)

        # Calculate the loss using the base loss function
        loss = self.base_loss(masked_pred, masked_target)

        return loss
