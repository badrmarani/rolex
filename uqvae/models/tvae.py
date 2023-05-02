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
        recon_x, log_scale = self.decoder(z)
        return recon_x, log_scale, mu, logvar

    def neg_log_likelihood(self, recon_x, x, log_scale):
        st = 0
        loss = []
        for column_info in output_info:
            for span_info in column_info:
                if span_info.activation_fn != 'softmax':
                    ed = st + span_info.dim
                    std = sigma[st]
                    eq = x[:, st] - torch.tanh(recon_x[:, st])
                    loss.append((eq ** 2 / 2 / (std ** 2)).sum())
                    loss.append(torch.log(std) * x.size()[0])
                    st = ed

                else:
                    ed = st + span_info.dim
                    loss.append(nn.functional.cross_entropy(
                        recon_x[:, st:ed], torch.argmax(x[:, st:ed], dim=-1), reduction='sum'))
                    st = ed

        return sum(loss) * factor / x.size()[0]

    def kl_divergence(self, mu, logvar):
        kld_loss = - 0.5 * \
            torch.sum(1 + logvar - mu.pow(2) - logvar.pow(2), dim=-1)
        return kld_loss

    def loss_function(self, recon_x, x, mu, logvar, log_scale):
        rec_loss = self.neg_log_likelihood(recon_x, x, log_scale)
        kld_loss = self.kl_divergence(mu, logvar)

        return (
            self.beta * rec_loss.mean(dim=0) + kld_loss.mean(dim=0),
            rec_loss.mean(dim=0),
            kld_loss.mean(dim=0),
        )
