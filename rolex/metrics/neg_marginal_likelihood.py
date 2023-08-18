import math
from typing import Optional

import torch
from torch import distributions, nn
from tqdm import trange

from .utils import enable_dropout


@torch.no_grad()
def neg_marginal_marginal_log_likelihood(
    encoder: nn.Module,
    decoder: nn.Module,
    n_samples_from_posterior: int,
    data: torch.Tensor,
    reduction: Optional[str] = "none",
) -> torch.Tensor:
    """Estimate the negative marginal log-likelihood using importance sampling method.

    Args:
        encoder (nn.Module): Encoder model.
        decoder (nn.Module): Decoder model.
        n_samples_from_posterior (int): The number of samples to draw.
        data (torch.Tensor): The input data for which the marginal log-likelihood is estimated.
        reduction (Optional[str], optional): Specifies the reduction to be applied to the estimated
            negative log-likelihood. Options are "none" (no reduction) and "mean" (mean reduction).
            Defaults to "none".

    Returns:
        torch.Tensor: Estimated negative marginal log-likelihood values. The shape of the
        returned tensor depends on the reduction option. If "none" is selected, the tensor
        has the same shape as the input data. If "mean" is selected, the tensor is a scalar.
    """
    encoder.eval()
    decoder.eval()
    enable_dropout(decoder)
    neg_logpx = []
    prior = distributions.Normal(0.0, 1.0)
    for l in range(n_samples_from_posterior):
        z_mu, z_logvar = encoder(data)
        z_std = z_logvar.mul(0.5).exp()
        qzx = distributions.Normal(z_mu, z_std)
        z = qzx.rsample()
        likelihood = distributions.Normal(*decoder(z)).log_prob(data).sum(-1)

        neg_logpx += [
            qzx.log_prob(z).sum(-1) - prior.log_prob(z).sum(-1) - likelihood
        ]

    neg_logpx = -math.log(n_samples_from_posterior) + torch.stack(
        neg_logpx, dim=1
    ).logsumexp(1)
    if reduction.lower() == "mean":
        return neg_logpx.mean()
    elif reduction.lower() == "none":
        return neg_logpx
    else:
        raise NotImplementedError(f"{reduction.lower()} is not implemented.")
