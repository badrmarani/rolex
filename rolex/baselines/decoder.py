from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from ..layers import BayesianLinear
from .mlp import make_block


class BayesianDecoder(nn.Module):
    """
    BayesianDecoder Module for decoding data using a sequence of Bayesian layers.

    Args:
        embedding_dim (float): The input embedding dimension.
        decompress_dims (Union[List[int], Tuple[int, ...]]): List or tuple of dimensions for decompression.
        data_dim (float): The output data dimension.
        dropout (float): Dropout probability.
        mode (Optional[str]): Mode for layer creation, e.g., "bayesian" (default: None).
        **kwargs: Additional keyword arguments to pass to the layers.

    Attributes:
        seq (nn.Sequential): Sequential module containing the layers.
        sigma (nn.Parameter): Learnable parameter for the standard deviation of the output distribution.

    """

    def __init__(
        self,
        embedding_dim: float,
        decompress_dims: Union[List[int], Tuple[int, ...]],
        data_dim: float,
        dropout: float,
        mode: Optional[str] = None,
        **kwargs,
    ) -> None:
        super(Decoder, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(decompress_dims):
            seq += make_block(dim, item, p=dropout, mode="bayesian", **kwargs)
            dim = item
        seq.append(BayesianLinear(dim, data_dim, **kwargs))
        self.seq = nn.Sequential(*seq)
        self.sigma = nn.Parameter(torch.ones(data_dim) * 0.1)

        self.embedding_dim = embedding_dim
        self.decompress_dims = decompress_dims
        self.data_dim = data_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass of the BayesianDecoder module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing the mean and standard deviation of the output distribution.
        """
        mu = self.seq(x)
        return mu, self.sigma.to(device=x.device, dtype=x.dtype)


class Decoder(nn.Module):
    """
    Decoder Module for decoding data using a sequence of linear layers.

    Args:
        embedding_dim (float): The input embedding dimension.
        decompress_dims (Union[List[int], Tuple[int, ...]]): List or tuple of dimensions for decompression.
        data_dim (float): The output data dimension.
        dropout (float): Dropout probability.
        mode (Optional[str]): Mode for layer creation, e.g., "bayesian" (default: None).
        **kwargs: Additional keyword arguments to pass to the layers.

    Attributes:
        seq (nn.Sequential): Sequential module containing the layers.
        sigma (nn.Parameter): Learnable parameter for the standard deviation of the output distribution.
    """

    def __init__(
        self,
        embedding_dim: float,
        decompress_dims: Union[List[int], Tuple[int, ...]],
        data_dim: float,
        dropout: float,
        mode: Optional[str] = None,
        **kwargs,
    ) -> None:
        super(Decoder, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(decompress_dims):
            seq += make_block(dim, item, p=dropout)
            dim = item
        seq.append(nn.Linear(dim, data_dim))
        self.seq = nn.Sequential(*seq)
        self.sigma = nn.Parameter(torch.ones(data_dim) * 0.1)

        self.embedding_dim = embedding_dim
        self.decompress_dims = decompress_dims
        self.data_dim = data_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass of the Decoder module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing the mean and standard deviation of the output distribution.
        """
        mu = self.seq(x)
        return mu, self.sigma.to(device=x.device, dtype=x.dtype)
