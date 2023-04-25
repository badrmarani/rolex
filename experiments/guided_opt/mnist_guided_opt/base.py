import torch
from torch import nn
from torch.utils import data

import os

import numpy as np

from utils import save_latent_space

class BaseVAE(nn.Module):
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def enable_dropout(self):
        for m in self.modules():
            if m.__class__.__name__.startswith("Dropout"):
                m.train()

    def rsample(self, mu, logvar):
        std = logvar.mul(0.5).exp()
        eps = torch.randn_like(std, device=std.device, dtype=std.dtype)
        return std.mul(eps).add(mu)
    
    @torch.no_grad()
    def generate(self, n_samples):
        self.decoder.eval()

        data = []
        steps = n_samples // self.batch_size+1
        for _ in range(steps):
            mean = torch.zeros(self.batch_size, self.lat_size)
            z = torch.normal(mean=mean, std=mean+1).to(self.device)
            fake, sigmas = self.decoder(z)
            fake = torch.tanh(fake)
            data.append(fake.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n_samples]
        if self.transform_data:
            return self.transformer.inverse_transform(data, sigmas.detach().cpu().numpy())
        else:
            return data

    def reconstruct(self, real):
        self.eval()
        if isinstance(real, data.DataLoader):
            real = real.dataset.data
        recon, mu, logvar, sigma = self.forward(real)
        recon = torch.tanh(recon)
        return recon.detach().cpu().numpy()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        mu, logvar = self.encoder(x)
        z = self.rsample(mu, logvar)
        xhat, sigma = self.decoder(z)
        return xhat, mu, logvar, sigma

    def train_one_epoch(
        self,
        epoch,
        loss_fn,
        optimizer,
        n_epochs,
        log_interval,
        save_model_every_x_epochs,
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
            loss, kld, rec = loss_fn(x, xhat, mu, logvar, sigma, 5.0)
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
                if plot_latent_space:
                    ls_savedir = os.path.join(save_imgs_path, "latent_space/") 
                    os.makedirs(ls_savedir, exist_ok=True)
                    if self.lat_size > 2:
                        pass
                    else:
                        plt = save_latent_space(
                            model=self,
                            data=self.val_loader,
                            n_samples=1000,
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

    def forward(self, x: torch.Tensor):
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
