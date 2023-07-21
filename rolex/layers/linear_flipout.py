import torch
from torch import nn

from .linear_variational import BayesianLinear


class LinearFlipout(BayesianLinear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_mean: float = 0,
        prior_variance: float = 1,
        posterior_mu_init: float = 0,
        posterior_rho_init: float = -3,
        bias: bool = True,
    ) -> None:
        """
        Implements Linear layer with Flipout reparameterization trick.
        Ref: https://arxiv.org/abs/1803.04386
        """
        super().__init__(
            in_features,
            out_features,
            prior_mean,
            prior_variance,
            posterior_mu_init,
            posterior_rho_init,
            bias,
        )

    def forward(self, inp):
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        weight_delta = weight_sigma * self.weight_eps.data.normal_()
        bias = None
        if self.bias_mu is not None:
            bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            bias = bias_sigma * self.bias_eps.data.normal_()
        out = nn.functional.linear(inp, self.weight_mu, self.bias_mu)

        sign_r = inp.clone().uniform_(-1, 1).sign()
        sign_s = out.clone().uniform_(-1, 1).sign()
        perturbed_outs = nn.functional.linear(inp * sign_r, weight_delta, bias) * sign_s
        return out + perturbed_outs
