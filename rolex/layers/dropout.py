import torch
from torch import nn


class DropConnect(nn.Dropout):
    """
    DropConnect regularization layer.

    Args:
        p (float): Probability of dropping a connection.

    Inherits from:
        nn.Dropout

    """

    def __init__(self, p: float) -> None:
        super(DropConnect, self).__init__()
        self.p = p

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the DropConnect layer.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying DropConnect regularization.
        """
        return nn.functional.dropout(
            input=input,
            p=self.p,
            training=self.training,
            inplace=self.inplace,
        ) * (1 - self.p)
