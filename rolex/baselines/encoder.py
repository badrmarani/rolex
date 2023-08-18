from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from .mlp import make_block


class Encoder(nn.Module):
    """
    Encoder Module for encoding input data into mean and log variance of the latent space.

    Args:
        data_dim (int): The input data dimension.
        compress_dims (Union[List[int], Tuple[int, ...]]): List or tuple of dimensions for compression.
        embedding_dim (int): The dimension of the embedding or latent space.
        **kwargs: Additional keyword arguments to pass to the layers.

    Attributes:
        seq (nn.Sequential): Sequential module containing the compression layers.
        fc1 (nn.Linear): Linear layer for mean computation.
        fc2 (nn.Linear): Linear layer for log variance computation.

    """

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
        """
        Forward pass of the Encoder module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing the mean and log variance of the latent space.
        """
        feat = self.seq(x)
        mu = self.fc1(feat)
        logvar = self.fc2(feat)
        return mu, logvar
