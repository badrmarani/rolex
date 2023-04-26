import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(
        self,
        data_dim,
        compress_dims,
        embedding_dim,
        add_dropouts=False,
        p=None,
    ):
        super().__init__()

        seq = []
        dim = data_dim
        for item in compress_dims:
            seq += [
                nn.Linear(dim, item), 
                nn.ReLU(),
            ]
            dim = item

            if add_dropouts:
                seq += [nn.Dropout(p)]

        self.seq = nn.Sequential(*seq)
        self.fc1 = nn.Linear(dim, embedding_dim)
        self.fc2 = nn.Linear(dim, embedding_dim)

    def forward(self, x: torch.Tensor):
        embeddings = self.seq(x)
        return (
            self.fc1(embeddings),
            self.fc2(embeddings),
        )


class Decoder(nn.Module):
    def __init__(
        self,
        embedding_dim,
        decompress_dims,
        data_dim,
        add_dropouts=False,
        p=None
    ):
        super().__init__()

        seq = []
        dim = embedding_dim
        for item in decompress_dims:
            seq += [
                nn.Linear(dim, item), 
                nn.ReLU(),
            ]
            dim = item
            if add_dropouts:
                seq += [nn.Dropout(p)]

        seq.append(nn.Linear(dim, data_dim))
        self.seq = nn.Sequential(*seq)
        self.sigma = nn.Parameter(torch.ones(data_dim) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x), self.sigma
