import torch
from torch import distributions, nn

from ..layers.bayesian import BayesianLinear


class BayesianKLLoss(nn.Module):
    def __init__(self, model: nn.Module):
        super(BayesianKLLoss, self).__init__()
        self.model = model

    def forward(self) -> torch.Tensor:
        kld = 0.0
        for m in self.model.modules():
            if isinstance(m, BayesianLinear):
                kld += distributions.kl_divergence(
                    distributions.Normal(m.weight_mu, torch.log1p(m.weight_rho.exp())),
                    distributions.Normal(m.prior_weight_mu, m.prior_weight_sigma),
                ).mean()
                if m.bias:
                    kld += distributions.kl_divergence(
                        distributions.Normal(m.bias_mu, torch.log1p(m.bias_rho.exp())),
                        distributions.Normal(m.prior_bias_mu, m.prior_bias_sigma),
                    ).mean()
        return kld
