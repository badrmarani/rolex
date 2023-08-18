import torch
from torch import nn


class BayesianLinear(nn.Module):
    """
    Bayesian Linear Layer with Gaussian Variational Distribution.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        prior_mean (float): Prior mean for weight initialization (default: 0.0).
        prior_variance (float): Prior variance for weight initialization (default: 1.0).
        posterior_mu_init (float): Initial value for posterior mean (default: 0).
        posterior_rho_init (float): Initial value for posterior rho (default: -3.0).
        bias (bool): Whether to include bias (default: True).

    Attributes:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        prior_mean (float): Prior mean for weight initialization.
        prior_variance (float): Prior variance for weight initialization.
        posterior_mu_init (float): Initial value for posterior mean.
        posterior_rho_init (float): Initial value for posterior rho.
        bias (bool): Whether bias is included.
        weight_mu (nn.Parameter): Weight mean parameter.
        weight_rho (nn.Parameter): Weight rho parameter.
        weight_eps (torch.Tensor): Weight epsilon for reparameterization.
        prior_weight_mu (torch.Tensor): Prior weight mean.
        prior_weight_sigma (torch.Tensor): Prior weight sigma.
        bias_mu (nn.Parameter | None): Bias mean parameter.
        bias_rho (nn.Parameter | None): Bias rho parameter.
        bias_eps (torch.Tensor | None): Bias epsilon for reparameterization.
        prior_bias_mu (torch.Tensor | None): Prior bias mean.
        prior_bias_sigma (torch.Tensor | None): Prior bias sigma.

    """

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
            "weight_eps",
            torch.Tensor(out_features, in_features),
            persistent=False,
        )
        self.register_buffer(
            "prior_weight_mu",
            torch.Tensor(out_features, in_features),
            persistent=False,
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
        """
        Initialize weight and bias parameters.
        """
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
        """
        Forward pass through the Bayesian Linear layer.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        weight_sigma = torch.log1p(self.weight_rho.exp())
        weight = self.weight_mu + weight_sigma * self.weight_eps.data.normal_()
        bias = None
        if self.bias_mu is not None:
            bias_sigma = torch.log1p(self.bias_rho.exp())
            bias = self.bias_mu + bias_sigma * self.bias_eps.data.normal_()
        out = nn.functional.linear(input, weight, bias)
        return out


class FlipoutLinear(BayesianLinear):
    """
    Bayesian Linear Layer with Flipout reparameterization trick.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        prior_mean (float): Prior mean for weight initialization (default: 0).
        prior_variance (float): Prior variance for weight initialization (default: 1).
        posterior_mu_init (float): Initial value for posterior mean (default: 0).
        posterior_rho_init (float): Initial value for posterior rho (default: -3).
        bias (bool): Whether to include bias (default: True).

    Inherits from:
        BayesianLinear

    """

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
        """
        Forward pass through the FlipoutLinear layer.

        Args:
            inp (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        weight_delta = weight_sigma * self.weight_eps.data.normal_()
        bias = None
        if self.bias_mu is not None:
            bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            bias = bias_sigma * self.bias_eps.data.normal_()
        out = nn.functional.linear(inp, self.weight_mu, self.bias_mu)

        sign_r = inp.clone().uniform_(-1, 1).sign()
        sign_s = out.clone().uniform_(-1, 1).sign()
        perturbed_outs = (
            nn.functional.linear(inp * sign_r, weight_delta, bias) * sign_s
        )
        return out + perturbed_outs
