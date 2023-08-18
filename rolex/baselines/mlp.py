from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from ..layers import BayesianLinear


def make_block(
    in_features: int,
    out_features: int,
    p: Optional[float] = 0.0,
    mode: Optional[str] = None,
    **kwargs,
) -> List[nn.Module]:
    """
    Create a block of layers.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        p (Optional[float]): Dropout probability (default: 0.0).
        mode (Optional[str]): Mode for layer creation, e.g., "bayesian" (default: None).
        **kwargs: Additional keyword arguments to pass to the layers.

    Returns:
        List[nn.Module]: List of layers composing the block.

    """
    if mode is not None and mode.lower() == "bayesian":
        linear = BayesianLinear
    else:
        linear = nn.Linear

    return [
        linear(in_features, out_features, **kwargs),
        nn.ReLU(),
        nn.Dropout(p),
    ]
