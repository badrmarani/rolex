import torch
from torch import nn, distributions

from .base import Encoder, Decoder, BaseVAE

class VAE(BaseVAE):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        configs,
    ):
        super().__init__()
        
        self.embedding_dim = configs["embedding_dim"]
        self.beta = configs["beta"]

        self.cat_columns = configs["cat_columns"]

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
        recon_x, logsigma = self.decoder(z)
        return recon_x, logsigma, mu, logvar

    def log_likelihood(self, recon_x, x, logsigma):
        batch_size = x.size(0)

        if self.cat_columns is not None:
            non_cat_columns = list(set(range(x.size(-1))) - set(self.cat_columns))

            rec_loss_ber = nn.functional.cross_entropy(
                torch.argmax(recon_x[:, self.cat_columns], dim=-1),
                x[:, self.cat_columns],
                reduction="none",
            )
        else:
            non_cat_columns = list(range(x.size(-1)))

        rec_loss = torch.tanh(recon_x[:, non_cat_columns]) - x[:, non_cat_columns]
        rec_loss = rec_loss.pow(2) / 2 / (logsigma.exp().pow(2)) + logsigma*x.size(0)

        if self.cat_columns is not None:
            rec_loss = torch.concatenate((rec_loss, rec_loss_ber), dim=-1)
        rec_loss = rec_loss.sum(dim=-1)
        return rec_loss

    def kl_divergence(self, mu, logvar):
        kld_loss = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.pow(2), dim=-1)
        return kld_loss
    
    def loss_function(self, recon_x, x, mu, logvar, sigma):
        rec_loss = self.log_likelihood(recon_x, x, sigma)
        kld_loss = self.kl_divergence(mu, logvar)
        
        return (
            self.beta * rec_loss.mean(dim=0) + kld_loss.mean(dim=0),
            rec_loss.mean(dim=0),
            kld_loss.mean(dim=0),
        )