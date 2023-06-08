import torch
from torch import distributions, nn


class ELBO(nn.Module):
    def __init__(self, beta: float) -> None:
        super(ELBO, self).__init__()
        self.beta = beta

    def forward(self, x: torch.Tensor, qzx: distributions, pxz: distributions):
        log_likelihood = pxz.log_prob(x).sum(-1).mean(-1)
        kld = (
            distributions.kl_divergence(qzx, distributions.Normal(0.0, 1.0))
            .sum(-1)
            .mean(-1)
        )
        loss = -(log_likelihood - kld * self.beta)
        return loss, -log_likelihood, kld * self.beta
