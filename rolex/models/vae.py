import torch
from torch import distributions, nn

from ..losses.elbo import ELBO
from .base import BaseVAE


class VAE(BaseVAE):
    def __init__(self, encoder, decoder, regressor=None) -> None:
        super().__init__(encoder, decoder, regressor)

        self.elbo = ELBO()

    def forward(
        self, data, device, betas=(1.0, 1.0)
    ):
        beta_kld, beta_reg = betas
        if len(data) == 2:
            x, y = data
            if len(y.size()) < 2:
                y.unsqueeze_(-1)
            y = y.to(device)
        else:
            (x,) = data
        x = x.to(device)

        z_mu, z_sigma = self.encoder(x)
        qzx = distributions.Normal(z_mu, z_sigma)

        z_sample = qzx.rsample()
        x_recon_mu, x_recon_sigma = self.decoder(z_sample)
        pxz = distributions.Normal(x_recon_mu, x_recon_sigma)

        prediciton_loss = torch.tensor(0)
        if self.regressor is not None:
            y_pred = self.regressor(z_sample)
            prediciton_loss = nn.functional.mse_loss(y, y_pred)

        log_likelihood, kld_loss = self.elbo(x, qzx, pxz)
        loss = -log_likelihood + beta_kld * kld_loss + beta_reg * prediciton_loss

        log = dict(
            rec=log_likelihood.item(),
            kld=kld_loss.item(),
            reg=prediciton_loss.item() if self.regressor is not None else None,
            loss=loss.item(),
        )

        return loss, log
