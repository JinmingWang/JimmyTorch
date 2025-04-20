import torch
import torch.nn as nn


class JimmyModel(nn.Module):
    """
    JimmyModel defines a model format that compatible with many different models.

    Most importantly, the forwardBackward function returns a dictionary, this is helpful for unifying the training code
    when you want to use different models for different datasets in your experiments.
    """

    def __init__(self):
        super(JimmyModel, self).__init__()
        self.loss_names = ["loss"]
        self.loss_fn = nn.CrossEntropyLoss()

    def forwardBackward(self, data_dict: dict, loss_scaler = None) -> tuple[dict, dict]:
        """
        Forward pass and backward pass of the model.
        :param data_dict: a dictionary containing the input data
        :param loss_scaler: a loss scaler for mixed precision training
        :return: a dictionary containing the loss and output
        """

        if loss_scaler is not None:
            with torch.autocast(device_type=data_dict['data'].device, dtype=torch.float16):
                output = self(data_dict['data'])
                loss = self.loss_fn(output, data_dict['target'])
            loss_scaler.scale(loss).backward()
        else:
            output = self(data_dict['data'])
            loss = self.loss_fn(output, data_dict['target'])
            loss.backward()

        return {"loss": loss.item()}, {"output": output.detach()}