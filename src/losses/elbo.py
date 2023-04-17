import torch
from torch import nn, distributions

class ELBOLoss(nn.Module):
    def __init__(self, device: torch.device) -> None:
        super().__init__()

        self.log_scale = nn.Parameter(
            torch.tensor([0.0], device=device),
            requires_grad=True,
        )
    
    def kld_divergence(self, z, mu, std):
        log_pz = distributions.Normal(
            torch.zeros_like(mu),
            torch.ones_like(std),
        ).log_prob(z)
        
        log_qzx = distributions.Normal(mu, std).log_prob(z)

        kld_loss = log_qzx - log_pz
        return kld_loss.sum(-1)

    def gaussian_likelihood(self, xhat, x):
        scale = self.log_scale.exp()
        log_pxz = distributions.Normal(xhat, scale).log_prob(x)
        return log_pxz.sum(-1)

    def bernoulli_likelihood(self, xhat, x):
        xhat = torch.sigmoid(xhat)
        log_pxz = distributions.Bernoulli(xhat, validate_args=False).log_prob(x)
        return log_pxz.sum(-1)

    def forward(self, x, xhat, mu, logvar):
        std = logvar.mul(0.5).exp()
        p_zx = distributions.Normal(mu, std)
        z = p_zx.rsample()
        
        kld_loss = self.kld_divergence(z, mu, std)
        rec_loss = self.bernoulli_likelihood(xhat, x)
        elbo_loss = kld_loss - rec_loss
        elbo_loss = elbo_loss.mean()
        
        return (
            elbo_loss,
            rec_loss.mean(),
            kld_loss.mean(),
        )
