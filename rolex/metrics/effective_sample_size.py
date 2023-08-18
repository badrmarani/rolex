from typing import Optional

import torch
from torch import distributions, nn
from tqdm import tqdm, trange

from .neg_marginal_likelihood import neg_marginal_marginal_log_likelihood
from .utils import enable_dropout


@torch.no_grad()
def ess_x(
    encoder: nn.Module,
    decoder: nn.Module,
    n_simulations: int,
    n_samples_from_posterior: int,
    data: torch.Tensor,
    reduction: Optional[str] = "none",
) -> torch.Tensor:
    """
    Calculate the Effective Sample Size of a sample from training data.

    Args:
        encoder (nn.Module): Encoder module.
        decoder (nn.Module): Decoder module.
        n_simulations (int): Number of simulations to perform.
        n_samples_from_posterior (int): Number of samples from the posterior.
        data (torch.Tensor): Input data.
        reduction (str, optional): Reduction strategy for the calculated Effective Sample Size (default: "none").

    Returns:
        torch.Tensor: Effective Sample Size of x. The shape of the
        returned tensor depends on the reduction option. If "none" is selected, the tensor
        has the same shape as the input data. If "mean" is selected, the tensor is a scalar.

    """
    log_ess_x = []
    for t in range(n_simulations):
        log_ess_x += [
            neg_marginal_marginal_log_likelihood(
                encoder=encoder,
                decoder=decoder,
                n_samples_from_posterior=n_samples_from_posterior,
                data=data,
                reduction="none",
            )
        ]
    log_ess_x = torch.stack(log_ess_x, dim=1)
    log_ess_x = 2 * (-log_ess_x + log_ess_x.sum(dim=-1).unsqueeze(-1))
    log_ess_x = -log_ess_x.logsumexp(1)
    if reduction.lower() == "mean":
        return log_ess_x.mean()
    elif reduction.lower() == "none":
        return log_ess_x
    else:
        raise NotImplementedError(f"{reduction.lower()} is not implemented.")


@torch.no_grad()
def effective_sample_size(
    encoder: nn.Module,
    decoder: nn.Module,
    n_simulations: int,
    n_samples_from_posterior: int,
    z_samples: torch.Tensor,
    reduction: Optional[str] = "none",
) -> torch.Tensor:
    """
    Calculate the Effective Sample Size of the latent space.

    Args:
        encoder (nn.Module): Encoder module.
        decoder (nn.Module): Decoder module.
        n_simulations (int): Number of simulations to perform.
        n_samples_from_posterior (int): Number of samples from the posterior.
        z_samples (torch.Tensor): Samples from the latent space.
        reduction (str, optional): Reduction strategy for the calculated Effective Sample Size (default: "none").

    Returns:
        torch.Tensor: Effective Sample Size of the latent space.

    """
    x, std = decoder(z_samples)
    x = torch.tanh(x)
    return ess_x(
        encoder=encoder,
        decoder=decoder,
        n_simulations=n_simulations,
        n_samples_from_posterior=n_samples_from_posterior,
        data=x,
        reduction=reduction,
    )
