import torch
from torch import distributions, nn

from ..layers import BayesianLinear


class BayesianELBOLoss(nn.Module):
    """
    Bayesian Evidence Lower Bound (ELBO) Loss.

    Args:
        model (nn.Module): Bayesian model for which the ELBO loss is calculated.

    Inherits from:
        nn.Module

    """

    def __init__(self, model: nn.Module):
        super(BayesianELBOLoss, self).__init__()
        self.model = model

    def forward(self) -> torch.Tensor:
        """
        Calculate the Bayesian ELBO loss.

        Returns:
            torch.Tensor: Bayesian ELBO loss.
        """
        kld = 0.0
        for m in self.model.modules():
            if isinstance(m, BayesianLinear):
                kld += distributions.kl_divergence(
                    distributions.Normal(
                        m.weight_mu, torch.log1p(m.weight_rho.exp())
                    ),
                    distributions.Normal(
                        m.prior_weight_mu, m.prior_weight_sigma
                    ),
                ).mean()
                if m.bias:
                    kld += distributions.kl_divergence(
                        distributions.Normal(
                            m.bias_mu, torch.log1p(m.bias_rho.exp())
                        ),
                        distributions.Normal(
                            m.prior_bias_mu, m.prior_bias_sigma
                        ),
                    ).mean()
        return kld
