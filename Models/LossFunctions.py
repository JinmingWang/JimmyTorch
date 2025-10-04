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

        mask = mask.to(torch.bool)

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
    def __init__(self, base_loss: Callable) -> None:
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


class RMSE(nn.Module):
    def __init__(self) -> None:
        """
        Initialize the RMSE loss function.
        """
        super(RMSE, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Calculate the Root Mean Square Error (RMSE).

        :param pred: the predicted output
        :param target: the target output
        :return: the RMSE loss
        """
        mse_loss = self.mse(pred, target)
        return torch.sqrt(mse_loss)


class KLDLoss(nn.Module):
    def __init__(self, free_bits: float = -6.0):
        """
        Initializes the VAELoss class.

        Parameters:
        - kl_weight (float): Weight to scale the KL divergence loss. Default is 1.0.
        """
        super(KLDLoss, self).__init__()
        # larger kl_weight
        # -> encourage smaller kl_loss
        # -> encourage larger torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp(), dim=-1)
        # -> encourage z_mean = 0
        # -> encourage 1 + z_logvar - z_logvar.exp() = 0
        # -> encourage z_logvar = 0
        # -> encourage exp(z_logvar) to be larger, so larger noise
        self.free_bits = free_bits

    def forward(self, z_mean: torch.Tensor, z_logvar: torch.Tensor):
        # Compute KL Divergence loss (KL divergence between N(z_mean, exp(z_logvar)) and N(0, 1))
        # To prevent numerical issues, we can clamp z_logvar to a reasonable range
        kl_loss = -0.5 * (1 + z_logvar - z_mean.pow(2) - z_logvar.exp())  # Per node
        kl_loss = torch.clamp(kl_loss, min=self.free_bits)
        kl_loss = kl_loss.sum(dim=-1).mean()  # Mean over batch

        return kl_loss