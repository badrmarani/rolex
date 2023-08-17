import torch
from torch import distributions
from tqdm import tqdm, trange
from utils import enable_dropout


@torch.no_grad()
def effective_sample_size(
    decoder,
    latent_sample,
    n_simulations,
    n_sampled_outcomes,
    dtype: torch.dtype = None,
    reduction="none",
    verbose=True,
):
    """
    Ref: https://arxiv.org/abs/1912.05651v3.
    """
    decoder.eval()
    enable_dropout(decoder)
    log_ess_x = []

    if verbose:
        iterator = trange(1, 1 + n_simulations, desc="Effective Sample Size")
    else:
        iterator = range(1, 1 + n_simulations)
    for _ in iterator:
        log_px = []

        x_recon_mu, x_recon_sigma = decoder(latent_sample)
        p_theta_0 = distributions.Normal(x_recon_mu, x_recon_sigma)
        x_recon = p_theta_0.rsample()
        for _ in range(n_sampled_outcomes):
            xx_recon_mu, xx_recon_sigma = decoder(latent_sample)
            p_theta_m = distributions.Normal(xx_recon_mu, xx_recon_sigma)
            log_px += [p_theta_m.log_prob(x_recon).sum(-1)]
        log_px = torch.stack(log_px, dim=1)
        log_wx = log_px - log_px.logsumexp(dim=1)[:, None]
        log_ess_x += [-torch.logsumexp(2 * log_wx, dim=1)]
    log_ess_x = torch.stack(log_ess_x, dim=1)
    log_ess_z = -torch.tensor(n_simulations).log() + log_ess_x.logsumexp(dim=-1)
    ess_z = log_ess_z.exp()
    if reduction.lower() == "none":
        return ess_z
    elif reduction.lower() == "mean":
        return ess_z.mean(-1)
    else:
        print(f"{reduction.lower()} is not implemented.")
