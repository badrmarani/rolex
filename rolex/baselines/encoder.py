from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from .mlp import make_block


class Encoder(nn.Module):
    def __init__(
        self,
        data_dim: int,
        compress_dims: Union[List[int], Tuple[int, ...]],
        embedding_dim: int,
        **kwargs,
    ) -> None:
        super(Encoder, self).__init__()

        dim = data_dim
        seq = []
        for item in list(compress_dims):
            seq += make_block(dim, item, p=0.0)
            dim = item

        self.seq = nn.Sequential(*seq)
        self.fc1 = nn.Linear(dim, embedding_dim)
        self.fc2 = nn.Linear(dim, embedding_dim)

        self.data_dim = data_dim
        self.compress_dims = compress_dims
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        feat = self.seq(x)
        mu = self.fc1(feat)
        logvar = self.fc2(feat)
        return mu, logvar
