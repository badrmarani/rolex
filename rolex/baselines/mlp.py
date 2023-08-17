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
    if mode is not None and mode.lower() == "bayesian":
        linear = BayesianLinear
    else:
        linear = nn.Linear

    return [
        linear(in_features, out_features, **kwargs),
        nn.ReLU(),
        nn.Dropout(p),
    ]
