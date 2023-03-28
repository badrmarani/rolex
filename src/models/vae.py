import torch
from torch import nn, distributions


class VAE(nn.Module):
    def __init__(
        self,
        inp_size: int,
        emb_size: int,
        lat_size: int,
        out_size: int,
    ):
        super(VAE, self).__init__()

        self.encoder = Encoder(inp_size, emb_size, lat_size)
        self.decoder = Decoder(lat_size, emb_size, out_size)

    def reparameterization(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        # qzx = distributions.Normal(mu, torch.exp(0.5*logvar))
        # z = qzx.rsample()
        return z

    def forward(self, tensor):
        mu, logvar = self.encoder(tensor)
        z = self.reparameterization(mu, logvar)
        xhat = self.decoder(z)
        return xhat, mu, logvar


class Encoder(nn.Module):
    def __init__(
        self,
        inp_size: int,
        emb_size: int,
        lat_size: int,
    ):
        super(Encoder, self).__init__()
        self.encode = nn.Sequential(
            nn.Linear(inp_size, emb_size),
            nn.Tanh(),
            nn.Linear(emb_size, emb_size),
            nn.Tanh(),
        )

        self.mu = nn.Linear(emb_size, lat_size)
        self.logvar = nn.Linear(emb_size, lat_size)

    def forward(self, tensor):
        tmp = self.encode(tensor)
        return (
            self.mu(tmp),
            self.logvar(tmp),
        )


class Decoder(nn.Module):
    def __init__(
        self,
        lat_size: int,
        emb_size: int,
        out_size: int,
    ):
        super(Decoder, self).__init__()
        self.decode = nn.Sequential(
            nn.Linear(lat_size, emb_size),
            nn.Tanh(),
            nn.Linear(emb_size, out_size),
        )

    def forward(self, tensor):
        return self.decode(tensor)
