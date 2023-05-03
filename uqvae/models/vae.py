import torch
from torch import nn, distributions

from .base import Encoder, Decoder, BaseVAE

class VAE(BaseVAE):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        model_config,
        categorical_columns=None,
    ):
        super().__init__()
        
        self.categorical_columns = categorical_columns
        self.beta = model_config.beta

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        self.encoder.train()
        self.decoder.train()
        mu, logvar = self.encoder(x)
        z = self.rsample(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

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

    def kl_divergence(self, mu, logvar):
        return - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.pow(2), dim=-1)
    
    def get_loss(self, recon_x, x, mu, logvar, epoch=None):
        rec_loss = self.reconstruction_loss(recon_x, x)
        kld_loss = self.kl_divergence(mu, logvar)
        
        return (
            self.beta * rec_loss.mean(dim=0) + kld_loss.mean(dim=0),
            rec_loss.mean(dim=0),
            kld_loss.mean(dim=0),
        )
