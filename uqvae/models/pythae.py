from typing import Optional
from pythae.models import VAMP
from pythae.models.nn import BaseDecoder, BaseEncoder
from pythae.models import VAMP, VAMPConfig
from pythae.models.base.base_utils import ModelOutput

from .base import Encoder, Decoder

import torch
from torch import nn

class BEncoder(BaseEncoder):
    def __init__(self, encoder: Encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        mu, logvar = self.encoder(x)
        return ModelOutput(
            embedding=mu,
            log_covariance=logvar,
        )


class BDecoder(BaseDecoder):
    def __init__(self, decoder: Decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, x):
        recon_x = self.decoder(x)
        return ModelOutput(
            reconstruction=recon_x
        )


class VAMP_(VAMP):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        model_config,
        categorical_columns,
    ):
        super().__init__(
            model_config=model_config,
            encoder=BEncoder(encoder),
            decoder=BDecoder(decoder),
        )
        
        self.categorical_columns = categorical_columns
    
    def forward(self, x):
        self.encoder.train()
        self.decoder.train()
        enc = self.encoder(x)
        mu, logvar = enc.embedding, enc.log_covariance
        z = mu + torch.randn_like(mu)*logvar.mul(0.5).exp()
        recon_x = self.decoder(z).reconstruction
        return recon_x, mu, logvar

    def get_loss(self, recon_x, x, mu, logvar, epoch):
        z = mu + torch.randn_like(mu)*logvar.mul(0.5).exp()
        return self.loss_function(recon_x, x, mu, logvar, z, epoch)

    def reconstruction_loss(self, recon_x, x):
        if self.categorical_columns is None:
            rec_loss = 0.5 * nn.functional.mse_loss(
                torch.tanh(recon_x),
                x,
                reduction="none",
            ).sum(dim=-1)
        else:
            non_categorical_columns = [i for i in range(x.size(-1)) if i not in self.categorical_columns]
            rec_loss = 0.5 * nn.functional.mse_loss(
                torch.tanh(recon_x[:, non_categorical_columns]),
                x[:, non_categorical_columns],
                reduction="none",
            ).sum(dim=-1)

            rec_loss += nn.functional.binary_cross_entropy(
                torch.softmax(recon_x[:, self.categorical_columns], dim=-1),
                x[:, self.categorical_columns],
                reduction="none",
            ).sum(dim=-1)
        return rec_loss

    def loss_function(self, recon_x, x, mu, log_var, z, epoch):
        recon_loss = self.reconstruction_loss(recon_x, x)

        log_p_z = self._log_p_z(z)

        log_q_z = (-0.5 * (log_var + torch.pow(z - mu, 2) /
                   log_var.exp())).sum(dim=1)
        KLD = -(log_p_z - log_q_z)

        if self.linear_scheduling > 0:
            beta = 1.0 * epoch / self.linear_scheduling
            if beta > 1 or not self.training:
                beta = 1.0

        else:
            beta = 1.0

        return (
            (recon_loss + beta * KLD).mean(dim=0),
            recon_loss.mean(dim=0),
            KLD.mean(dim=0),
        )

    def _log_p_z(self, z):
        """Computation of the log prob of the VAMP"""

        C = self.number_components

        x = self.pseudo_inputs(self.idle_input.to(z.device)).reshape(
            (C,) + self.model_config.input_dim
        )

        # we bound log_var to avoid unbounded optim
        encoder_output = self.encoder(x)
        prior_mu, prior_log_var = (
            encoder_output.embedding,
            encoder_output.log_covariance,
        )

        z_expand = z.unsqueeze(1)
        prior_mu = prior_mu.unsqueeze(0)
        prior_log_var = prior_log_var.unsqueeze(0)

        log_p_z = (
            torch.sum(
                -0.5
                * (
                    prior_log_var
                    + (z_expand - prior_mu) ** 2 / torch.exp(prior_log_var)
                ),
                dim=2,
            )
            - torch.log(torch.tensor(C).type(torch.float))
        )

        log_p_z = torch.logsumexp(log_p_z, dim=1)

        return log_p_z

