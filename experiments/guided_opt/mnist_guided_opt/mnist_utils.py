import os
import pickle
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from ctgan.data_transformer import DataTransformer
from torch import nn
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import MNIST


def transform_data(self, data, data_discrete_cols):
    transformer = DataTransformer()
    transformer.fit(data, data_discrete_cols)


@torch.no_grad()
def plot_latent_space(model: nn.Module, data: data.DataLoader, n_samples: int, device: torch.device):
    plt.figure()
    y = data.dataset.targets[:n_samples].cpu().to(torch.float)
    x = data.dataset.data[:n_samples, ...].flatten(1).to(torch.float) / 255.
    z = model.rsample(*model.encoder(x)).cpu()
    plt.scatter(z[:, 0], z[:, 1], c=y, cmap="tab10")
    plt.colorbar()
    return plt


# def get_mnist_loaders(root, batch_size, transform_data) -> Tuple[data.DataLoader]:
#     fit_dataset = MNIST(root, train=True, download=True)
#     val_dataset = MNIST(root, train=False, download=True)
#     if transform_data:
#         print("transforming data...")
#         columns = [str(x) for x in range(fit_dataset.data.size(-1)**2)]
#         tmp_fit_dataset = pd.DataFrame(fit_dataset.data.flatten(start_dim=1), columns=columns)
#         tmp_val_dataset = pd.DataFrame(val_dataset.data.flatten(start_dim=1), columns=columns)
#         transformer = DataTransformer()
#         transformer.fit(tmp_fit_dataset, ())
#         trans_fit_dataset = transformer.transform(tmp_fit_dataset)
#         trans_val_dataset = transformer.transform(tmp_val_dataset)
#         fit_dataset = data.TensorDataset(torch.from_numpy(trans_fit_dataset.astype("float32")))
#         val_dataset = data.TensorDataset(torch.from_numpy(trans_val_dataset.astype("float32")))

#     fit_loader = data.DataLoader(dataset=fit_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True,)

#     return fit_loader, val_loader


def loss_function(
    x: torch.Tensor,
    xhat: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    sigma: torch.Tensor,
    beta: float = 1.0,
    transform_info=None,
) -> Tuple[torch.Tensor]:
    rec_loss = []
    if transform_info is not None:
        for column_info in transform_info:
            for span_info in column_info:
                if span_info.activation_fn != "softmax":
                    ed = st + span_info.dim
                    std = sigma[st]
                    eq = x[:, st] - torch.tanh(xhat[:, st])
                    rec_loss.append((eq ** 2 / 2 / (std ** 2)).sum())
                    rec_loss.append(torch.log(std) * x.size()[0])
                    st = ed
                else:
                    ed = st + span_info.dim
                    rec_loss.append(nn.functional.cross_entropy(
                        xhat[:, st:ed], torch.argmax(x[:, st:ed], dim=-1), reduction="sum"))
                    st = ed
    else:
        eq = x - torch.tanh(xhat)
        rec_loss = eq**2/2/(sigma**2) + sigma.log()*x.size(0)
        rec_loss = rec_loss.sum(dim=-1)

    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.pow(2), dim=-1)
    loss = kld_loss.mean(dim=0) + beta * rec_loss.mean(dim=0)
    return (
        loss,
        kld_loss.mean(dim=0),
        rec_loss.mean(dim=0),
    )


def LDE(log_a, log_b):
    max_log = torch.max(log_a, log_b)
    min_log = torch.min(log_a, log_b)
    return max_log + torch.log(1 + torch.exp(min_log - max_log))


@torch.no_grad()
def mutual_information_importance_sampling(
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
        device = torch.device("cpu")
    z1_space = torch.arange(
        -grid_size, grid_size, step_size, dtype=torch.float, device=device
    )
    z2_space = torch.arange(
        -grid_size, grid_size, step_size, dtype=torch.float, device=device
    )
    z1, z2 = torch.meshgrid(z1_space, z2_space, indexing="xy")

    mi = mutual_information_importance_sampling(
        model.to(device),
        torch.stack((z1.flatten(), z2.flatten()), dim=1),
        n_simulations=n_simulations,
        n_sampled_outcomes=n_sampled_outcomes,
    ).view(z1.size())

    plt.figure()
    # plt.contourf(
    #     z1.detach().cpu(),
    #     z2.detach().cpu(),
    #     mi.detach().cpu(),
    # )
    plt.imshow(mi.detach().cpu(), cmap="autumn", interpolation="nearest")

    plt.colorbar()
    return plt


class AuxNetwork(nn.Module):
    def __init__(
        self,
        inp_size: int,
        emb_sizes: Dict[str, List[int]],
        add_dropouts: bool,
    ) -> None:
        super().__init__()

        self.seq = nn.ModuleList()
        pre_emb_size = inp_size
        for block in emb_sizes.values():
            s = []
            for i, n_features in enumerate(block):
                s += [nn.Linear(pre_emb_size, n_features)]
                if i < len(block) - 1:
                    s += [nn.ReLU()]

                    if add_dropouts:
                        s += [nn.Dropout(0.2)]

                pre_emb_size = n_features        
            s = nn.Sequential(*s)
            self.seq.append(s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        out1 = self.seq[0](x)
        out2 = self.seq[1](out1)
        return torch.log_softmax(out2, dim=1)


class Encoder(nn.Module):
    def __init__(
        self,
        inp_size,
        emb_sizes,
        lat_size,
        add_dropouts,
        p: float = None,
    ):
        super().__init__()

        seq = []
        pre_emb_size = inp_size
        for i, emb_size in enumerate(emb_sizes):
            seq += [nn.Linear(pre_emb_size, emb_size), nn.ReLU()]
            pre_emb_size = emb_size

            if add_dropouts:
                seq += [nn.Dropout(p)]

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
    def __init__(
        self,
        lat_size,
        emb_sizes,
        out_size,
        add_dropouts,
        p: float = 0.2
    ):
        super().__init__()

        seq = []
        inv_emb_sizes = emb_sizes[::-1] + [out_size]
        pre_emb_size = lat_size
        for i, emb_size in enumerate(inv_emb_sizes):
            seq += [nn.Linear(pre_emb_size, emb_size)]
            if i < len(inv_emb_sizes) - 1:
                seq += [nn.ReLU()]
                if add_dropouts:
                    seq += [nn.Dropout(p)]

            pre_emb_size = emb_size

        self.seq = nn.Sequential(*seq)
        self.sigma = nn.Parameter(torch.ones(out_size) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x), self.sigma


class GuidedVAE(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        guide_latent_space: bool,
        inp_size: int,
        n_gradient_steps: int = None,
        n_simulations: int = None,
        n_sampled_outcomes: int = None,
        gradient_scale: float = None,
        uncertainty_threshold_value: float = None,
        normalize_gradient: bool = None,
        aux_network: nn.Module = None,
        transform_data: bool = None,
        save_transformed_data: bool = None,
        save_transformed_path: str = None,
        batch_size: int = None,
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.aux_network = aux_network

        self.inp_size = inp_size

        self.n_gradient_steps = n_gradient_steps
        self.n_simulations = n_simulations
        self.n_sampled_outcomes = n_sampled_outcomes
        self.normalize_gradient = normalize_gradient
        self.gradient_scale = gradient_scale
        self.uncertainty_threshold_value = uncertainty_threshold_value
        self.guide_latent_space = guide_latent_space
        self.save_transformed_data = save_transformed_data
        self.save_transformed_path = save_transformed_path
        self.transform_data = transform_data
        self.batch_size = batch_size

    def get_mnist_loaders(self, root) -> Tuple[data.DataLoader]:
        if self.transform_data:
            fit_dataset = MNIST(root, train=True, download=True)
            val_dataset = MNIST(root, train=False, download=True)
            print("transforming data...")
            columns = [str(x) for x in range(fit_dataset.data.size(-1)**2)]
            tmp_fit_dataset = pd.DataFrame(fit_dataset.data.flatten(start_dim=1), columns=columns)
            tmp_val_dataset = pd.DataFrame(val_dataset.data.flatten(start_dim=1), columns=columns)
            self.transformer = DataTransformer()
            self.transformer.fit(tmp_fit_dataset, ())
            trans_fit_dataset = self.transformer.transform(tmp_fit_dataset)
            trans_val_dataset = self.transformer.transform(tmp_val_dataset)
            fit_dataset = data.TensorDataset(torch.from_numpy(trans_fit_dataset.astype("float32")).to("cpu"))
            val_dataset = data.TensorDataset(torch.from_numpy(trans_val_dataset.astype("float32")).to("cpu"))
            if self.save_transformed_data:
                os.makedirs(self.save_transformed_path, exist_ok=True)
                torch.save(fit_dataset, os.path.join(
                    self.save_transformed_path, "transformed_fit_dataset.pkl"
                ))
                torch.save(val_dataset, os.path.join(
                    self.save_transformed_path, "transformed_val_dataset.pkl"
                ))
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(torch.flatten),
            ])
            fit_dataset = MNIST(root, train=True, download=True, transform=transform)
            val_dataset = MNIST(root, train=False, download=True, transform=transform)

        fit_loader = data.DataLoader(dataset=fit_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = data.DataLoader(dataset=val_dataset, batch_size=self.batch_size, shuffle=True)
        return fit_loader, val_loader

    def enable_dropout(self):
        for m in self.modules():
            if m.__class__.__name__.startswith("Dropout"):
                m.train()

    def rsample(self, mu, logvar):
        std = logvar.mul(0.5).exp()
        eps = torch.randn_like(std, device=std.device, dtype=std.dtype)
        return std.mul(eps).add(mu)

    @torch.no_grad()
    def generate(self, n_samples, device):
        self.decoder.eval()
        
        data = []
        steps = n_samples // self.batch_size+1
        for _ in range(steps):
            mean = torch.zeros(self.batch_size, self.lat_size)
            z = torch.normal(mean=mean, std=mean+1).to(device)
            fake, sigmas = self.decoder(z)
            fake = torch.tanh(fake)
            data.append(fake.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n_samples]
        return self.transformer.inverse_transform(data, sigmas.detach().cpu().numpy())
    

    def forward(self, x) -> Tuple[torch.Tensor]:
        x = x.view(x.size(0), -1)
        mu, logvar = self.encoder(x)
        z = self.rsample(mu, logvar)
        if self.guide_latent_space:
            z = self.gradient_ascent_optimisation(z)
        xhat, sigma = self.decoder(z)
        return xhat, mu, logvar, sigma

    def gradient_ascent_optimisation(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        for _ in range(self.n_gradient_steps):
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
            mi = mutual_information_importance_sampling(self, z)

            mask = mi <= self.uncertainty_threshold_value
            mask = mask.unsqueeze(-1).repeat(1, z.size(-1))
            updated_z = torch.where(mask, updated_z, z)

        return updated_z
