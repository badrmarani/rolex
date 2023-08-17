import torch
from torch import distributions
from tqdm import tqdm, trange
from utils import enable_dropout, lde


@torch.no_grad()
def mutual_information(
    decoder,
    latent_sample,
    n_simulations,
    n_sampled_outcomes,
    dtype: torch.dtype = None,
    reduction="none",
    verbose=True,
):
    decoder.eval()
    enable_dropout(decoder)
    log_mi = []
    if verbose:
        iterator = trange(1, n_simulations + 1, desc="Mutual Information")
    else:
        iterator = range(1, n_simulations + 1)
    for _ in iterator:
        log_psm = []
        x_recon_mu, x_recon_sigma = decoder(latent_sample)
        p_theta_0 = distributions.Normal(x_recon_mu, x_recon_sigma)
        x_recon = p_theta_0.rsample()
        for _ in range(n_sampled_outcomes):
            xx_recon_mu, xx_recon_sigma = decoder(latent_sample)
            p_theta_m = distributions.Normal(xx_recon_mu, xx_recon_sigma)
            log_psm += [p_theta_m.log_prob(x_recon).mean(-1)]
        log_psm = torch.stack(log_psm, dim=1)
        log_psm = torch.where(log_psm <= 0, log_psm, -log_psm)

        log_ps = -torch.tensor(n_sampled_outcomes).log() + torch.logsumexp(
            log_psm, dim=1
        )
        log_hs_left = -torch.tensor(n_sampled_outcomes).log() + torch.logsumexp(
            log_psm + torch.log(-log_psm), dim=1
        )

        log_hs_right = log_ps + torch.log(-log_ps)
        log_hs = lde(log_hs_left, log_hs_right)
        log_mi += [log_hs - log_ps]
    log_mi = torch.stack(log_mi, dim=1)
    log_mi_avg = -torch.tensor(n_simulations).log() + torch.logsumexp(log_mi, dim=1)
    mi = log_mi_avg.exp()
    if reduction.lower() == "none":
        return mi
    elif reduction.lower() == "mean":
        return mi.mean(-1)
    else:
        print(f"{reduction.lower()} is not implemented.")
