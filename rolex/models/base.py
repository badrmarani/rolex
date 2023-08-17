from argparse import ArgumentParser
from typing import Tuple

import pytorch_lightning as pl
import torch
from torch import distributions, nn

from ..losses import BayesianELBOLoss
from ..utils.parser import parse_list


class BaseVAE(pl.LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        lr: float,
        weight_decay: float,
        beta_on_kld: float,
        bayesian_decoder: str,
        **kwargs,
    ) -> None:
        super(BaseVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.lr = lr
        self.weight_decay = weight_decay
        self.beta_on_kld = beta_on_kld
        self.bayesian_decoder = bayesian_decoder
        if (
            self.bayesian_decoder is not None
            and self.bayesian_decoder.lower() == "bayesian"
        ):
            self.bayes_elbo_loss_fn = BayesianELBOLoss(model=self.decoder)

        self.automatic_optimization = False

    @classmethod
    def add_model_specific_args(
        cls,
        parser: ArgumentParser,
    ) -> ArgumentParser:
        parser.register("type", list, parse_list)
        parser.add_argument("--compress_dims", type=list, default="[128,128]")
        parser.add_argument("--decompress_dims", type=list, default="[128,128]")
        parser.add_argument("--embedding_dim", type=int, default=2)
        parser.add_argument("--dropout", type=float, default=0.2)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--weight_decay", type=float, default=1e-5)
        parser.add_argument("--beta_on_kld", type=float, default=2.0)
        parser.add_argument("--bayesian_decoder", type=str, default=None)
        return parser

    def configure_optimizers(self) -> torch.optim.Optimizer:
        opt = torch.optim.Adam(
            params=list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return opt

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        mu, logvar = self.encoder(x)
        std = logvar.mul(0.5).exp() + 1e-10
        z = mu + torch.randn_like(std) * std
        return z, mu, logvar

    def decode(self, z: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        mu, std = self.decoder(z)
        return mu, std

    def reconstruction_loss_fn(
        self,
        x: torch.Tensor,
        recon_x: torch.Tensor,
        sigmas: torch.Tensor,
    ) -> torch.Tensor:
        decoder_distributions = distributions.Normal(recon_x, sigmas)
        neg_log_likelihood = -decoder_distributions.log_prob(x)
        return neg_log_likelihood.sum() / x.size(0)

    def kld_loss_fn(
        self,
        z: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        loss = 0.5 * (logvar.exp() + mu.pow(2) - 1.0 - logvar)
        loss = loss.sum() / z.size(0)
        return loss

    def forward(self, batch: Tuple[torch.Tensor, ...]):
        if len(batch) == 2:
            x, y = batch
        else:
            (x,) = batch

        z, mu, logvar = self.encode(x)
        recon_x, std = self.decode(z)

        kld_loss = self.kld_loss_fn(z, mu, logvar)
        rec_loss = self.reconstruction_loss_fn(x, recon_x, std)

        bayes_kld_loss = 0.0
        if (
            self.bayesian_decoder is not None
            and self.bayesian_decoder.lower() == "bayesian"
        ):
            bayes_kld_loss = self.bayes_elbo_loss_fn()

        loss = rec_loss + kld_loss * self.beta_on_kld + bayes_kld_loss

        if self.logging_prefix is not None:
            self.log(
                f"rec/{self.logging_prefix}", rec_loss, prog_bar=self.log_progress_bar
            )
            self.log(
                f"kld/{self.logging_prefix}", kld_loss, prog_bar=self.log_progress_bar
            )
            self.log(
                f"loss/{self.logging_prefix}", loss, prog_bar=self.log_progress_bar
            )
        return loss

    def training_step(
        self, batch: Tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor:
        self.log("beta_on_kld", self.beta_on_kld, prog_bar=False)
        self.logging_prefix = "train"
        self.log_progress_bar = False

        opt = self.optimizers()
        opt.zero_grad()
        loss = self(batch)
        self.manual_backward(loss)
        opt.step()
        self.decoder.sigma.data.clamp_(0.01, 1.0)

        self.log_progress_bar = None
        self.logging_prefix = None

    def validation_step(
        self, batch: Tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor:
        self.logging_prefix = "val"
        self.log_progress_bar = True

        loss = self(batch)

        self.log_progress_bar = None
        self.logging_prefix = False
        return loss
