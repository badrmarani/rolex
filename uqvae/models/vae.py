import torch
from torch import nn, distributions

from .base import Encoder, Decoder, BaseVAE

class VAE(BaseVAE):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        configs,
        categorical_columns=None,
    ):
        super().__init__()
        
        self.categorical_columns = categorical_columns
        self.embedding_dim = configs["embedding_dim"]
        self.beta = configs["beta"]
        self.configs = configs

        self.encoder = encoder(
            configs["data_dim"],
            configs["compress_dims"],
            configs["embedding_dim"],
        )

        self.decoder = decoder(
            configs["embedding_dim"],
            configs["decompress_dims"],
            configs["data_dim"],
        )

    def forward(self, x):
        self.encoder.train()
        self.decoder.train()
        mu, logvar = self.encoder(x)
        z = self.rsample(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

    def reconstruction_loss(self, recon_x, x):
        return 0.5 * nn.functional.mse_loss(
            torch.tanh(recon_x),
            x,
            reduction="none",
        ).sum(dim=-1)

    def kl_divergence(self, mu, logvar):
        kld_loss = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.pow(2), dim=-1)
        return kld_loss
    
    def loss_function(self, recon_x, x, mu, logvar):
        rec_loss = self.reconstruction_loss(recon_x, x)
        kld_loss = self.kl_divergence(mu, logvar)
        
        return (
            self.beta * rec_loss.mean(dim=0) + kld_loss.mean(dim=0),
            rec_loss.mean(dim=0),
            kld_loss.mean(dim=0),
        )
