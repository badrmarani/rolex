import torch
from torch import distributions
from tqdm import trange
from utils import enable_dropout


@torch.no_grad()
def marginal_log_likelihood_estimator(
    model,
    data,
    n_sampled_outcomes,
    dtype: torch.dtype = None,
    reduction="none",
    verbose=True,
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

        qzx = distributions.Normal(*model.encoder(data))
        z = qzx.rsample()
        pxz = distributions.Normal(*model.decoder(z))

        log_p += [
            qzx.log_prob(z).sum(-1)
            - prior.log_prob(z).sum(-1)
            - pxz.log_prob(data).sum(-1)
        ]
    log_p = torch.stack(log_p, dim=-1)
    log_p = torch.tensor(data.size(0)).log() - log_p.logsumexp(1)
    p = log_p.exp()
    if reduction.lower() == "none":
        return p
    elif reduction.lower() == "mean":
        return p.mean(-1)
    else:
        print(f"{reduction.lower()} is not implemented.")
