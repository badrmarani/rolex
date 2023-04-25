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

from base import BaseVAE
from utils import save_latent_space, loss_function

def lde(log_a, log_b):
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

        tmp_log_hs = lde(left_log_hs, right_log_hs) - log_ps
        log_mi.append(tmp_log_hs)

    log_mi = torch.stack(log_mi, dim=1)
    log_mi_avg = -torch.log(torch.tensor(n_simulations).float()) + torch.logsumexp(
        log_mi, dim=1
    )

    return log_mi_avg.exp()


def save_mutual_information(
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
        lat_size: int,
        emb_sizes: List,
        add_dropouts: bool,
        p: float = None,
    ) -> None:
        super().__init__()

        seq = []
        pre_emb_size = lat_size
        for i, es in enumerate(emb_sizes):
            seq += [nn.Linear(pre_emb_size, es)]
            if i<len(emb_sizes)-1:
                seq += [nn.ReLU()]
                if add_dropouts:
                    seq += [nn.Dropout(p)]
            pre_emb_size = es
                    
        self.seq = nn.Sequential(*seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.seq(x)


class GuidedVAE(BaseVAE):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        inp_size: int,
        device: torch.device,
        batch_size: int,
        lat_size: int,
        loss_beta: float = 1.0,
        guide_latent_space: bool = False,
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
    ) -> None:
        super(GuidedVAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.aux_network = aux_network

        self.inp_size = inp_size
        self.lat_size = lat_size
        self.batch_size = batch_size
        self.device = device
        self.loss_beta = loss_beta

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

    def get_mnist_loaders(self, root) -> Tuple[data.DataLoader]:
        if self.transform_data:
            fit_dataset = MNIST(root, train=True, download=True)
            val_dataset = MNIST(root, train=False, download=True)
            print("transforming data...")
            columns = [str(x) for x in range(fit_dataset.data.size(-1)**2)]
            tmp_fit_dataset = pd.DataFrame(fit_dataset.data.flatten(start_dim=1), columns=columns)
            tmp_val_dataset = pd.DataFrame(val_dataset.data.flatten(start_dim=1), columns=columns)
            self.transformer = DataTransformer()
            self.transformer.fit(tmp_fit_dataset.sample(1000), ())
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

        self.fit_loader = data.DataLoader(dataset=fit_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = data.DataLoader(dataset=val_dataset, batch_size=self.batch_size, shuffle=True)
        return self.fit_loader, self.val_loader

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

    def train_one_epoch(
        self,
        epoch,
        loss_fn,
        optimizer,
        n_epochs,
        log_interval,
        save_model_every_x_epochs,
        guide_latent_space,
        save_imgs_path,
        odirname,
        plot_latent_space,
    ):
        self.train()
        running_loss = 0.0
        for i, batch in enumerate(self.fit_loader, 1):
            x, y = batch[0].to(self.device), batch[1].to(self.device)
            optimizer.zero_grad()
            xhat, mu, logvar, sigma = self(x)
            loss, kld, rec = loss_fn(x, xhat, mu, logvar, sigma, torch.tensor(self.loss_beta, device=self.device))
            loss.backward()
            optimizer.step()
            self.decoder.sigma.data.clamp_(0.01, 1.0)

            running_loss += loss.item()
            if i % log_interval == log_interval - 1:
                print(
                    "[{}/{}, {}/{}] training loss: {:.3f}".format(
                        epoch,
                        n_epochs,
                        i,
                        len(self.fit_loader),
                        running_loss / log_interval,
                    )
                )
                running_loss = 0.0
        
        with torch.no_grad():
            if not epoch%save_model_every_x_epochs:
                self.eval()
                if guide_latent_space:
                    grid_size = 20
                    step_size = 0.5
                    plt = save_mutual_information(
                        model=self,
                        grid_size=grid_size,
                        step_size=step_size,
                        n_simulations=self.n_simulations,
                        n_sampled_outcomes=self.n_sampled_outcomes,
                        device=self.device,
                    )
                    
                    mi_savedir = os.path.join(save_imgs_path, "mutual_information/") 
                    os.makedirs(mi_savedir, exist_ok=True)
                    plt.savefig(
                        os.path.join(mi_savedir, "plot_mi_epoch_{}.jpg".format(epoch)),
                        dpi=300
                    )
                if plot_latent_space:
                    ls_savedir = os.path.join(save_imgs_path, "latent_space/") 
                    os.makedirs(ls_savedir, exist_ok=True)
                    if self.lat_size > 2:
                        pass
                    else:
                        plt = save_latent_space(
                            model=self,
                            data=self.val_loader,
                            n_samples=None,
                            device=self.device,
                        )

                    plt.savefig(
                        os.path.join(ls_savedir, "plot_ls_epoch_{}.jpg".format(epoch)),
                        dpi=300
                    )

                weights_dirname = os.path.join(odirname, "weights")
                os.makedirs(weights_dirname, exist_ok=True)
                torch.save(
                    {
                        "state_dict": self.to("cpu").state_dict(),
                    },
                    os.path.join(weights_dirname, "mnist_vae_{}_weights.pkl".format(
                        epoch,
                    )),
                )

        self.to(self.device)
