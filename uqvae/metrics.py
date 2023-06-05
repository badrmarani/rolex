import torch
from torch import nn, distributions

from tqdm import trange

def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()


def lde(log_a, log_b):
    max_log = torch.max(log_a, log_b)
    min_log = torch.min(log_a, log_b)
    return max_log + torch.log(1 + torch.exp(min_log - max_log))


@torch.no_grad()
def mutual_information(
    decoder,
    latent_sample,
    n_simulations,
    n_sampled_outcomes,
    verbose=True,
):
    decoder.train()
    log_mi = []
    if verbose:
        mrange = trange(1, n_simulations+1, desc=f"mutual_information")
    else:
        mrange = range(1, n_simulations+1)
    for s in mrange:
        log_psm = []
        p_theta_0 = decoder(latent_sample)
        x_recon = p_theta_0.rsample()
        for m in range(n_sampled_outcomes):
            p_theta_m = decoder(latent_sample)
            log_psm += [p_theta_m.log_prob(x_recon).mean(-1)]
        log_psm = torch.stack(log_psm, dim=1)
        log_psm = torch.where(log_psm <= 0, log_psm, -log_psm)

        log_ps = (
            - torch.tensor(n_sampled_outcomes, dtype=torch.float32).log()
            + torch.logsumexp(log_psm, dim=1)
        )

        log_hs_left = (
            - torch.tensor(n_sampled_outcomes, dtype=torch.float32).log()
            + torch.logsumexp(log_psm + torch.log(-log_psm), dim=1)
        )

        log_hs_right = log_ps + torch.log(-log_ps)
        log_hs = lde(log_hs_left, log_hs_right)
        log_mi += [log_hs - log_ps]
    log_mi = torch.stack(log_mi, dim=1)
    log_mi_avg = (
        - torch.tensor(n_simulations, dtype=torch.float32).log()
        + torch.logsumexp(log_mi, dim=1)
    )
    return log_mi_avg.exp()
