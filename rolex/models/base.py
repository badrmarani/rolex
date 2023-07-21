import torch
from torch import distributions, nn

from ..layers.bayesian import BayesianLinear


class MLPRegressor(nn.Module):
    def __init__(self, embedding_dim) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.ELU(alpha=0.6),
            nn.Linear(embedding_dim * 4, embedding_dim * 4),
            nn.ELU(alpha=0.6),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim * 4, 1),
            nn.Linear(1, 1, bias=False),
        )

    def forward(self, x):
        return self.seq(x)


class Encoder(nn.Module):
    def __init__(self, data_dim, compress_dims, embedding_dim, bayesian=False):
        super(Encoder, self).__init__()

        if bayesian:
            linear = BayesianLinear
        else:
            linear = nn.Linear

        dim = data_dim
        seq = []
        for item in list(compress_dims):
            seq += [linear(dim, item), nn.ELU(alpha=0.6)]
            dim = item

        self.seq = nn.Sequential(*seq)
        self.fc1 = linear(dim, embedding_dim)
        self.fc2 = linear(dim, embedding_dim)

    def forward(self, x):
        feat = self.seq(x)
        mu = self.fc1(feat)
        sigma = self.fc2(feat).mul(0.5).exp()
        return mu, sigma


class Decoder(nn.Module):
    def __init__(self, embedding_dim, decompress_dims, data_dim, p=0.4, bayesian=False):
        super(Decoder, self).__init__()

        if bayesian:
            linear = BayesianLinear
            p = 0.0
        else:
            linear = nn.Linear

        dim = embedding_dim
        seq = []
        for item in list(decompress_dims):
            seq += [linear(dim, item), nn.Dropout(p), nn.ELU(alpha=0.6)]
            dim = item
        seq.append(linear(dim, data_dim))
        self.seq = nn.Sequential(*seq)
        self.logvar = nn.Parameter(torch.zeros(data_dim))

    def forward(self, x):
        mu = self.seq(x)
        sigma = self.logvar.mul(0.5).exp()
        return mu, sigma


class BaseVAE(nn.Module):
    def __init__(self, encoder, decoder, regressor=None) -> None:
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.regressor = regressor

    def configure_optimizer(self, config):
        parameters = set()
        parameters |= set(self.encoder.parameters())
        parameters |= set(self.decoder.parameters())
        if self.regressor is not None:
            parameters |= set(self.regressor.parameters())
        optimizer = torch.optim.Adam(
            parameters, lr=config.lr, weight_decay=config.weight_decay
        )

        scheduler = None
        if config.add_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, "min", factor=0.2, patience=1, min_lr=config.lr
            )
        return optimizer, scheduler
