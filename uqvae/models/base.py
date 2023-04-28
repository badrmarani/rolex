import torch
from torch import nn

class BaseVAE(nn.Module):
    def __init__(self):
        super(BaseVAE, self).__init__()

    def rsample(self, mu, logvar):
        std = logvar.mul(0.5).exp()
        return mu + std * torch.randn_like(std) 
    
    @torch.no_grad()
    def reconstruct(self, x):
        self.encoder.eval()
        self.decoder.eval()
        z = self.rsample(*self.encoder(x))
        recon_x, sigmas = self.decoder(z)
        return recon_x, sigmas

    @torch.no_grad()
    def generate(self, n_samples, device, batch_size=32):
        self.decoder.eval()

        data = []
        steps = n_samples // batch_size + 1
        for _ in range(steps):
            mean = torch.zeros(batch_size, self.embedding_dim)
            z = torch.normal(mean=mean, std=mean+1).to(device)
            recon_x, sigmas = self.decoder(z)
            recon_x = torch.tanh(recon_x)
            data.append(recon_x.detach().cpu().numpy())

        data = torch.concatenate(data, axis=0)[:n_samples]
        return data

    def loss_function(self):
        raise NotImplementedError


class Encoder(nn.Module):
    def __init__(
        self,
        data_dim,
        compress_dims,
        embedding_dim,
        add_dropouts=False,
        p=None,
        is_conditioned=False,
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

            if add_dropouts:
                seq += [nn.Dropout(p)]

        self.dim = dim
        self.seq = nn.Sequential(*seq)
        if not is_conditioned:
            self.fc1 = nn.Linear(self.dim, embedding_dim)
            self.fc2 = nn.Linear(self.dim, embedding_dim)

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
        p=None,
        **kwargs,
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
            if add_dropouts:
                seq += [nn.Dropout(p)]

        seq.append(nn.Linear(dim, data_dim))
        self.seq = nn.Sequential(*seq)
        self.logsigma = nn.Parameter(torch.ones(data_dim) * (-2.3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x), self.logsigma
