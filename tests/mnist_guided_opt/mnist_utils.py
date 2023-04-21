from typing import List, Tuple

import torch
from torch import nn
from torch.utils import data

from torchvision.datasets import MNIST
from torchvision import transforms

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["backend"] = "Qt5Agg"


def get_mnist_loaders(batch_size) -> Tuple[data.DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.Lambda(torch.flatten),
        ]
    )
    fit_loader = data.DataLoader(
        dataset=MNIST(
            "tests/mnist_guided_opt/mnist_dataset",
            train=True,
            download=True,
            transform=transform,
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = data.DataLoader(
        dataset=MNIST(
            "tests/mnist_guided_opt/mnist_dataset",
            train=False,
            download=True,
            transform=transform,
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    return fit_loader, val_loader


def loss_function(
    x: torch.Tensor,
    xhat: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> Tuple[torch.Tensor]:
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.pow(2), dim=-1)
    rec_loss = 0.5 * nn.functional.mse_loss(
        xhat.flatten(1),
        x.flatten(1),
        reduction="none",
    ).sum(-1)

    loss = beta * kld_loss + rec_loss

    return (
        loss.mean(0),
        kld_loss.mean(0),
        rec_loss.mean(0),
    )


def LDE(log_a, log_b):
    max_log = torch.max(log_a, log_b)
    min_log = torch.min(log_a, log_b)
    return max_log + torch.log(1 + torch.exp(min_log - max_log))


@torch.no_grad()
def mutual_information_is(
    network,
    z: torch.Tensor,
    n_simulations: int = 10,
    n_sampled_outcomes: int = 10,
) -> torch.Tensor:
    log_mi = []

    network.train()
    for s in range(n_simulations):
        all_log_psm = []

        xhat = network.decoder(z)

        for _ in range(n_sampled_outcomes):
            network.eval()
            network.enable_dropout()

            xxhat = network.decoder(z)
            log_psm = -nn.functional.gaussian_nll_loss(
                xxhat,
                xhat,
                torch.ones_like(xhat, device=xhat.device),
                reduction="none",
                full=True,
            ).sum(-1)

            all_log_psm.append(log_psm)

        all_log_psm = torch.stack(all_log_psm, dim=1)
        log_ps = -torch.log(torch.tensor(n_sampled_outcomes).float()) + torch.logsumexp(
            all_log_psm, dim=1
        )

        right_log_hs = log_ps + torch.log(-log_ps)
        psm_log_psm = all_log_psm + torch.log(-all_log_psm)
        left_log_hs = -torch.log(
            torch.tensor(n_sampled_outcomes).float()
        ) + torch.logsumexp(psm_log_psm, dim=1)

        tmp_log_hs = LDE(left_log_hs, right_log_hs) - log_ps
        log_mi.append(tmp_log_hs)

    log_mi = torch.stack(log_mi, dim=1)
    log_mi_avg = -torch.log(torch.tensor(n_simulations).float()) + torch.logsumexp(
        log_mi, dim=1
    )

    return log_mi_avg.exp()


def plot_mutual_information(
    model,
    grid_size,
    step_size,
    n_simulations,
    n_sampled_outcomes,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z1_space = torch.arange(
        -grid_size, grid_size, step_size, dtype=torch.float, device=device
    )
    z2_space = torch.arange(
        -grid_size, grid_size, step_size, dtype=torch.float, device=device
    )
    z1, z2 = torch.meshgrid(z1_space, z2_space, indexing="xy")

    mi = mutual_information_is(
        model,
        torch.stack((z1.flatten(), z2.flatten()), dim=1),
        n_simulations=n_simulations,
        n_sampled_outcomes=n_sampled_outcomes,
    ).view(z1.size())

    plt.figure()
    plt.contourf(
        z1.detach().cpu(),
        z2.detach().cpu(),
        mi.detach().cpu(),
    )

    plt.colorbar()
    plt.show()


class AuxNetwork(nn.Module):
    def __init__(self, inp_size: int, emb_sizes: List[int]) -> None:
        super().__init__()

        seq = []
        pre_emb_size = inp_size
        for i, emb_size in enumerate(emb_sizes):
            seq += [nn.Linear(pre_emb_size, emb_size)]
            if i < len(emb_sizes) - 1:
                seq += [nn.ReLU()]

            pre_emb_size = emb_size
        self.seq = nn.Sequential(*seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return torch.log_softmax(self.seq(x), dim=1)


class Encoder(nn.Module):
    def __init__(self, inp_size, emb_sizes, lat_size, add_dropouts):
        super().__init__()

        seq = []
        pre_emb_size = inp_size
        for i, emb_size in enumerate(emb_sizes):
            seq += [nn.Linear(pre_emb_size, emb_size), nn.ReLU()]
            pre_emb_size = emb_size

            if add_dropouts:
                seq += [nn.Dropout(0.5)]

        self.seq = nn.Sequential(*seq)

        self.mu = nn.Linear(pre_emb_size, lat_size)
        self.logvar = nn.Linear(pre_emb_size, lat_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        embeddings = self.seq(x)
        return (
            self.mu(embeddings),
            self.logvar(embeddings),
        )


class Decoder(nn.Module):
    def __init__(self, lat_size, emb_sizes, out_size, add_dropouts):
        super().__init__()

        seq = []
        inv_emb_sizes = emb_sizes[::-1] + [out_size]
        pre_emb_size = lat_size
        for i, emb_size in enumerate(inv_emb_sizes):
            seq += [nn.Linear(pre_emb_size, emb_size)]
            if i < len(inv_emb_sizes) - 1:
                seq += [nn.ReLU()]
                if add_dropouts:
                    seq += [nn.Dropout(0.5)]

            pre_emb_size = emb_size

        self.seq = nn.Sequential(*seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class GuidedVAE(nn.Module):
    def __init__(
        self,
        inp_size: int,
        emb_sizes: int,
        lat_size: int,
        n_gradient_steps: int,
        n_simulations: int,
        n_sampled_outcomes: int,
        gradient_scale: float,
        uncertainty_threshold_value: float,
        add_dropouts: bool,
        normalize_gradient: bool,
        aux_network: nn.Module,
    ) -> None:
        super().__init__()

        self.encoder = Encoder(inp_size, emb_sizes, lat_size, add_dropouts)
        self.decoder = Decoder(lat_size, emb_sizes, inp_size, add_dropouts)
        self.aux_network = aux_network

        self.n_gradient_steps = n_gradient_steps
        self.n_simulations = n_simulations
        self.n_sampled_outcomes = n_sampled_outcomes
        self.normalize_gradient = normalize_gradient
        self.gradient_scale = gradient_scale
        self.uncertainty_threshold_value = uncertainty_threshold_value

    def enable_dropout(self):
        for m in self.modules():
            if m.__class__.__name__.startswith("Dropout"):
                m.train()

    def rsample(self, mu, logvar):
        std = logvar.mul(0.5).exp()
        eps = torch.randn_like(std, device=std.device, dtype=std.dtype)
        return std.mul(std).add(mu)

    @torch.no_grad()
    def reconstruct(self, tensor):
        return self.forward(tensor)[0]

    @torch.no_grad()
    def generate(self, n_samples, device):
        z = torch.randn((n_samples, self.lat_size), device=device)
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        x = x.view(x.size(0), -1)
        mu, logvar = self.encoder(x)
        z = self.gradient_ascent_optimisation(z=self.rsample(mu, logvar), x=x)
        xhat = self.decoder(z)
        return xhat, mu, logvar

    def gradient_ascent_optimisation(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        for step in range(self.n_gradient_steps):
            P = self.aux_network(z)
            gradient = torch.autograd.grad(
                outputs=P,
                inputs=z,
                grad_outputs=torch.ones_like(P, device=z.device),
                retain_graph=False,
            )[0]

            if self.normalize_gradient:
                gradient /= gradient.norm(2)

            updated_z = z + gradient * self.gradient_scale
            mi = mutual_information_is(z)

            mask = mi <= self.uncertainty_threshold_value
            mask = mask.unsqueeze(-1).repeat(1, 2)
            updated_z = torch.where(mask, updated_z, z)

        return updated_z
