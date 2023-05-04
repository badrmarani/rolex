import torch
from torch import nn

from src.pythae.models.base.base_utils import ModelOutput
from src.pythae.models.nn import BaseDecoder, BaseEncoder

class Encoder(BaseEncoder):
    def __init__(
        self,
        data_dim,
        compress_dims,
        embedding_dim,
    ):
        super(Encoder, self).__init__()


        seq = []
        dim = data_dim
        for item in compress_dims:
            seq += [
                nn.Linear(dim, item), 
                nn.ReLU(),
            ]
            dim = item

        self.dim = dim
        self.seq = nn.Sequential(*seq)
        self.fc1 = nn.Linear(self.dim, embedding_dim)
        self.fc2 = nn.Linear(self.dim, embedding_dim)

    def forward(self, x: torch.Tensor):
        embeddings = self.seq(x)
        return ModelOutput(
            embedding=self.fc1(embeddings),
            log_covariance=self.fc2(embeddings),
        )


class Decoder(BaseDecoder):
    def __init__(
        self,
        embedding_dim,
        decompress_dims,
        data_dim,
    ):
        super(Decoder, self).__init__()

        seq = []
        dim = embedding_dim
        for item in decompress_dims:
            seq += [
                nn.Linear(dim, item), 
                nn.ReLU(),
            ]
            dim = item

        seq.append(nn.Linear(dim, data_dim))
        self.seq = nn.Sequential(*seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ModelOutput(reconstruction=self.seq(x))
