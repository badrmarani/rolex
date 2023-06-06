import torch
from torch import distributions, nn


class Encoder(nn.Module):
    def __init__(self, data_dim, compress_dims, embedding_dim):
        super(Encoder, self).__init__()
        dim = data_dim
        seq = []
        for item in list(compress_dims):
            seq += [nn.Linear(dim, item), nn.ReLU()]
            dim = item

        self.seq = nn.Sequential(*seq)
        self.fc1 = nn.Linear(dim, embedding_dim)
        self.fc2 = nn.Linear(dim, embedding_dim)

    def forward(self, x):
        feat = self.seq(x)
        mu = self.fc1(feat)
        logvar = self.fc2(feat)
        normal = distributions.Normal(mu, logvar.mul(0.5).exp())
        normal.loc = normal.loc.to(x)
        normal.scale = normal.scale.to(x)
        return normal


class Decoder(nn.Module):
    def __init__(self, embedding_dim, decompress_dims, data_dim):
        super(Decoder, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(decompress_dims):
            seq += [nn.Linear(dim, item), nn.Dropout(0.2), nn.ReLU()]
            dim = item
        seq.append(nn.Linear(dim, data_dim))
        self.seq = nn.Sequential(*seq)
        self.logvar = nn.Parameter(torch.zeros(data_dim))

    def forward(self, x):
        mu = self.seq(x)
        normal = distributions.Normal(mu, self.logvar.mul(0.5).exp())
        normal.loc = normal.loc.to(x)
        normal.scale = normal.scale.to(x)
        return normal
