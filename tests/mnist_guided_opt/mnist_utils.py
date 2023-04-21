from typing import List

import torch
from torch import nn


class AuxNetwork(nn.Module):
    def __init__(self, inp_size: int, emb_sizes: List[int]) -> None:
        super().__init__()

        seq = []
        pre_emb_size = inp_size
        for i, emb_size in enumerate(emb_sizes):
            seq += [nn.Linear(pre_emb_size, emb_size)]
            if i < len(emb_sizes) - 1:
                seq += [nn.ReLU()]

            pre_emb_size = emb_size
        self.seq = nn.Sequential(*seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return torch.log_softmax(self.seq(x), dim=1)
