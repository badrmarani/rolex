import torch
from torch import nn, distributions


class ELBOLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, xhat, mu, logvar, mode):
        if mode.lower() == "gaussian":
            # x|z ~ N(xhat,1)
            # rec_loss = nn.functional.mse_loss(xhat, x, reduciton="sum")
            # rec_loss = (x-xhat)**2 + torch.tensor(2*torch.pi).log()
            # rec_loss *= - 0.5
            # rec_loss = rec_loss.mean()
            rec_loss = nn.functional.gaussian_nll_loss(
                xhat,
                x,
                var=torch.ones_like(x),
                full=False,
                reduction="sum",
            )
        else:
            # x|z ~ Ber(xhat)
            rec_loss = nn.functional.binary_cross_entropy_with_logits(
                xhat, x, reduciton="sum"
            )

        kld_loss = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())

        # pz = distributions.Normal(
        #     torch.zeros_like(mu),
        #     torch.ones_like(mu),
        # )

        # qz = distributions.Normal(mu, torch.exp(0.5*logvar))
        # kld_loss = - distributions.kl_divergence(qz, pz).sum()

        # rec_loss = distributions.Normal(
        #     xhat,
        #     0.01 * torch.ones_like(xhat),
        # ).log_prob(x).sum(-1).mean()
        elbo_loss = rec_loss + kld_loss

        return (
            elbo_loss,
            rec_loss,
            kld_loss,
        )
