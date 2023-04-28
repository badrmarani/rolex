import torch
from torch import nn
from torch.utils import data

from ctgan.data_transformer import DataTransformer

from .base import Encoder, Decoder
from .vae_benchmark.src.pythae.models.nn import BaseEncoder, BaseDecoder
from .vae_benchmark.src.pythae.models.base.base_utils import ModelOutput
from .vae_benchmark.src.pythae.models import VAMP, VAMPConfig

import numpy as np

class TVAMP(nn.Module):
    def __init__(
        self,
        train_data,
        embedding_dim=128,
        compress_dims=(128, 128),
        decompress_dims=(128, 128),
        l2scale=1e-5,
        batch_size=500,
        epochs=300,
        loss_factor=2,
        cuda=True,
        number_components = 50,
        discrete_columns=(),
        transformer=None,
        transform_data=True,
    ) -> None:
        super().__init__()

        self.transform_data = transform_data

        if self.transform_data:
            if transformer is not None:
                self.transformer = transformer
            else:
                self.transformer = DataTransformer()
                self.transformer.fit(train_data, discrete_columns)
            
            self.train_data = self.transformer.transform(train_data)
            self.data_dim = self.transformer.output_dimensions
            self.output_info = self.transformer.output_info_list
        else:
            self.train_data = train_data.values
            self.data_dim = train_data.shape[-1]
            self.output_info=None

        self.embedding_dim = embedding_dim
        self.compress_dims = compress_dims
        self.decompress_dims = decompress_dims

        self.l2scale = l2scale
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss_factor = loss_factor
        self.device = torch.device("cuda" if cuda else "cpu")

        self.number_components = number_components
        self.pseudo_inputs = nn.Sequential(
            nn.Linear(self.number_components, int(self.data_dim)),
            nn.Hardtanh(0.0, 1.0),
        ).to(self.device)
        self.idle_input = torch.eye(
            self.number_components, requires_grad=False
        ).to(self.device)

    def loss_function(
        self,
        x_recon,
        x,
        z,
        sigma,
        mu,
        logvar,
        output_info,
        loss_factor,
    ):
        if self.transform_data:
            st = 0
            loss = []
            for column_info in output_info:
                for span_info in column_info:
                    if span_info.activation_fn != "softmax":
                        ed = st + span_info.dim
                        std = sigma[st]
                        eq = x[:, st] - torch.tanh(x_recon[:, st])
                        loss.append((eq ** 2 / 2 / (std ** 2)).sum())
                        loss.append(torch.log(std) * x.size()[0])
                        st = ed

                    else:
                        ed = st + span_info.dim
                        loss.append(nn.functional.cross_entropy(
                            x_recon[:, st:ed], torch.argmax(x[:, st:ed], dim=-1), reduction="sum"))
                    st = ed
            assert st == x_recon.size()[1]
            recon_loss = sum(loss)
        else:
            recon_loss = nn.functional.mse_loss(
                x_recon.view(x.size(0), -1),
                x.view(x.size(0), -1),
                reduction="none",
            ).sum(dim=-1)
            
        log_p_z = self._log_p_z(z)

        log_q_z = (-0.5 * (logvar + torch.pow(z - mu, 2) / logvar.exp())).sum(dim=1)
        KLD = -(log_p_z - log_q_z)

        return (
            (recon_loss + loss_factor * KLD).mean(dim=0),
            torch.mean(recon_loss, dim=0),
            KLD.mean(dim=0) / loss_factor,
        )
        
    def _log_p_z(self, z):
        C = self.number_components

        x = self.pseudo_inputs(self.idle_input.to(z.device)).reshape(
            (C,) + (self.data_dim,)
        )

        # we bound log_var to avoid unbounded optim
        mu, logvar = self.encoder(x)
        prior_mu, prior_log_var = (
            mu, logvar,
        )

        z_expand = z.unsqueeze(1)
        prior_mu = prior_mu.unsqueeze(0)
        prior_log_var = prior_log_var.unsqueeze(0)

        log_p_z = (
            torch.sum(
                -0.5
                * (
                    prior_log_var
                    + (z_expand - prior_mu) ** 2 / torch.exp(prior_log_var)
                ),
                dim=2,
            )
            - torch.log(torch.tensor(C).type(torch.float))
        )

        log_p_z = torch.logsumexp(log_p_z, dim=1)

        return log_p_z

    def fit(self):
        # build the model with pythae
        self.encoder = Encoder(self.data_dim, self.compress_dims, self.embedding_dim).to(self.device)
        self.decoder = Decoder(self.embedding_dim, self.decompress_dims, self.data_dim).to(self.device)

        dataset = torch.from_numpy(self.train_data.astype("float32")).to(self.device)
        dataset = data.TensorDataset(dataset)
        loader = data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            weight_decay=self.l2scale,
        )


        for i in range(1, self.epochs+1):
            for idx, batch in enumerate(loader):
                optimizer.zero_grad()
                x = batch[0].to(self.device)
                mu, logvar = self.encoder(x)
                z = logvar.mul(0.5).exp()
                x_recon, sigma = self.decoder(z)
                loss, loss_1, loss_2 = self.loss_function(
                    x_recon=x_recon,
                    x=x,
                    z=z,
                    sigma=sigma,
                    mu=mu,
                    logvar=logvar,
                    output_info=self.output_info,
                    loss_factor=self.loss_factor,
                )
                # loss = loss_1 + loss_2
                loss.backward()
                optimizer.step()
                self.decoder.sigma.data.clamp_(0.01, 1.0)
            print("epoch {} loss: {:.3f}".format(i, loss))
