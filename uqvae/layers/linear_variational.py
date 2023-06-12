import torch
from torch import nn


class BayesianLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_mean: float = 0.0,
        prior_variance: float = 1.0,
        posterior_mu_init: float = 0,
        posterior_rho_init: float = -3.0,
        bias: bool = True,
    ) -> None:
        super(BayesianLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init
        self.bias = bias

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer(
            "weight_eps", torch.Tensor(out_features, in_features), persistent=False
        )
        self.register_buffer(
            "prior_weight_mu", torch.Tensor(out_features, in_features), persistent=False
        )
        self.register_buffer(
            "prior_weight_sigma",
            torch.Tensor(out_features, in_features),
            persistent=False,
        )

        if self.bias:
            self.bias_mu = nn.Parameter(torch.Tensor(out_features))
            self.bias_rho = nn.Parameter(torch.Tensor(out_features))
            self.register_buffer(
                "bias_eps", torch.Tensor(out_features), persistent=False
            )
            self.register_buffer(
                "prior_bias_mu", torch.Tensor(out_features), persistent=False
            )
            self.register_buffer(
                "prior_bias_sigma", torch.Tensor(out_features), persistent=False
            )
        else:
            self.register_buffer("prior_bias_mu", None, persistent=False)
            self.register_buffer("prior_bias_sigma", None, persistent=False)
            self.register_buffer("bias_mu", None, persistent=False)
            self.register_buffer("bias_sigma", None, persistent=False)
            self.register_buffer("bias_eps", None, persistent=False)

        self.init_parameters()

    def init_parameters(self) -> None:
        self.prior_weight_mu.fill_(self.prior_mean)
        self.prior_weight_sigma.fill_(self.prior_variance)
        self.weight_mu.data.normal_(mean=self.posterior_mu_init, std=0.1)
        self.weight_rho.data.normal_(mean=self.posterior_mu_init, std=0.1)
        if self.bias is not None:
            self.prior_bias_mu.fill_(self.prior_mean)
            self.prior_bias_sigma.fill_(self.prior_variance)
            self.bias_mu.data.normal_(mean=self.posterior_mu_init, std=0.1)
            self.bias_rho.data.normal_(mean=self.posterior_rho_init, std=0.1)

    def forward(self, input):
        weight_sigma = torch.log1p(self.weight_rho.exp())
        weight = self.weight_mu + weight_sigma * self.weight_eps.data.normal_()
        bias = None
        if self.bias_mu is not None:
            bias_sigma = torch.log1p(self.bias_rho.exp())
            bias = self.bias_mu + bias_sigma * self.bias_eps.data.normal_()
        out = nn.functional.linear(input, weight, bias)
        return out
