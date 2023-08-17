import torch
from torch import nn


class DropConnect(nn.Dropout):
    def __init__(self, p: float) -> None:
        super(DropConnect, self).__init__()
        self.p = p

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return nn.functional.dropout(
            input=input,
            p=self.p,
            training=self.training,
            inplace=self.inplace,
        ) * (1 - self.p)
