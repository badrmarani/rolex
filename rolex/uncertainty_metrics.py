import torch
from torch import distributions, nn
from tqdm import trange

from .utils import enable_dropout, lde


@torch.no_grad()
def mutual_information(
    decoder,
    latent_sample,
    n_simulations,
    n_sampled_outcomes,
    verbose=True,
):
    decoder.eval()
    enable_dropout(decoder)
    log_mi = []
    if verbose:
        iterator = trange(1, n_simulations + 1, desc=f"Mutual Information")
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

        log_ps = -torch.tensor(
            n_sampled_outcomes, dtype=torch.float32
        ).log() + torch.logsumexp(log_psm, dim=1)

        log_hs_left = -torch.tensor(
            n_sampled_outcomes, dtype=torch.float32
        ).log() + torch.logsumexp(log_psm + torch.log(-log_psm), dim=1)

        log_hs_right = log_ps + torch.log(-log_ps)
        log_hs = lde(log_hs_left, log_hs_right)
        log_mi += [log_hs - log_ps]
    log_mi = torch.stack(log_mi, dim=1)
    log_mi_avg = -torch.tensor(
        n_simulations, dtype=torch.float32
    ).log() + torch.logsumexp(log_mi, dim=1)
    return log_mi_avg.exp()


@torch.no_grad()
def effective_sample_size(
    decoder,
    latent_sample,
    n_simulations,
    n_sampled_outcomes,
    reduction="none",
    verbose=True,
):
    """
    Ref: https://arxiv.org/abs/1912.05651v3.
    """
    decoder.eval()
    enable_dropout(decoder)
    log_ess_x = []
    iterator = trange if verbose else range
    for _ in iterator(n_simulations):
        log_px = []
        p0 = decoder(latent_sample)
        x = p0.rsample()
        for _ in range(n_sampled_outcomes):
            pi = decoder(latent_sample)
            log_px += [pi.log_prob(x).sum(-1)]
        log_px = torch.stack(log_px, dim=1)
        log_wx = log_px - log_px.logsumexp(dim=1)[:, None]
        log_ess_x += [-torch.logsumexp(2 * log_wx, dim=1)]
    log_ess_x = torch.stack(log_ess_x, dim=1)
    log_ess_z = -torch.tensor(n_simulations).log() + log_ess_x.logsumexp(dim=-1)
    if reduction.lower() == "none":
        return log_ess_z
    elif reduction.lower() == "mean":
        return log_ess_z.mean(-1)
    else:
        print(f"{reduction.lower()} is not implemented.")


@torch.no_grad()
def marginal_log_likelihood_estimator(
    model, data, n_sampled_outcomes=1000, reduction="none", verbose=True
):
    """
    Estimate the marginal log-likelihood using importance sampling method.
    Ref: Appendix D of the paper, https://arxiv.org/abs/1312.6114.
    """
    model.eval()
    enable_dropout(model)
    log_p = []
    prior = distributions.Normal(0.0, 1.0)
    iterator = trange if verbose else range
    for _ in iterator(n_sampled_outcomes):
        qzx, pxz = model(data)
        z = qzx.rsample()
        log_p += [
            qzx.log_prob(z).sum(-1)
            - prior.log_prob(z).sum(-1)
            - pxz.log_prob(data).sum(-1)
        ]
    log_p = torch.stack(log_p, dim=-1)
    log_p = torch.tensor(data.size(0)).log() - log_p.logsumexp(1)
    if reduction.lower() == "none":
        return log_p
    elif reduction.lower() == "mean":
        return log_p.mean(-1)
    else:
        print(f"{reduction.lower()} is not implemented.")
