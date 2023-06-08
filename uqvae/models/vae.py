import argparse
import itertools
from typing import Any, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torchvision.utils import make_grid

from ..losses.elbo import ELBO
from ..metrics import ContrastiveLoss
from ..utils import parse_list
from .base import Decoder, Encoder


class BaseVAE(pl.LightningModule):
    def __init__(self, hparams, data_dim):
        super().__init__()

        self.data_dim = data_dim
        hparams.data_dim = data_dim
        self.save_hyperparameters(hparams)
        self.embedding_dim = hparams.embedding_dim

        self.elbo = ELBO()
        self.set_encoder_decoder()

        self.logging_prefix = None
        self.log_progress_bar = False

    def set_encoder_decoder(self):
        self.encoder = Encoder(
            self.data_dim, self.hparams.compress_dims, self.hparams.embedding_dim
        )
        self.decoder = Decoder(
            self.hparams.embedding_dim, self.hparams.decompress_dims, self.data_dim
        )

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, "min", factor=0.2, patience=1, min_lr=self.hparams.lr
            ),
            "interval": "epoch",
            "monitor": "loss/val",
        }
        return dict(optimizer=opt, lr_scheduler=scheduler)

    def training_step(self, batch, batch_idx):
        self.log("beta", self.hparams.beta, prog_bar=True)

        self.logging_prefix = "train"
        self.log_progress_bar = True
        self.log_progress_bar = True
        loss = self(batch)
        self.logging_prefix = None
        self.log_progress_bar = False
        return loss

    def validation_step(self, batch, batch_idx):
        self.logging_prefix = "val"
        self.log_progress_bar = True
        loss = self(batch)
        self.logging_prefix = None
        self.log_progress_bar = False
        return loss

    def forward(self, batch):
        if not self.hparams.semi_supervised_learning:
            (x,) = batch
        else:
            x, y = batch
            y = y.view(-1, 1)
        qzx = self.encoder(x)
        z = qzx.rsample()
        pxz = self.decoder(z)
        log_likelihood, kld = self.elbo(x, qzx, pxz)
        loss = log_likelihood + kld * self.hparams.beta

        metric_loss = 0.0
        if self.hparams.semi_supervised_learning:
            contrastive_loss = ContrastiveLoss(
                threshold=self.hparams.metric_loss_threshold,
                hard=self.hparams.metric_loss_hard,
            )
            metric_loss = contrastive_loss(z, y)
            loss += metric_loss * self.hparams.metric_loss_beta

        if self.logging_prefix is not None:
            self.log(
                f"rec/{self.logging_prefix}",
                log_likelihood,
                prog_bar=self.log_progress_bar,
            )
            self.log(f"kl/{self.logging_prefix}", kld, prog_bar=self.log_progress_bar)
            if self.hparams.semi_supervised_learning:
                self.log(
                    f"metric_loss:contrastive/{self.logging_prefix}",
                    metric_loss,
                    prog_bar=self.log_progress_bar,
                )
            self.log(f"loss/{self.logging_prefix}", loss)

        return loss

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.register("type", list, parse_list)
        vae_group = parser.add_argument_group("VAE")
        vae_group.add_argument("--metric_loss_hard", default=False, action="store_true")
        vae_group.add_argument("--metric_loss_threshold", type=float, required=False)
        vae_group.add_argument("--metric_loss_beta", type=float, required=False)
        vae_group.add_argument("--embedding_dim", type=int, required=True)
        vae_group.add_argument(
            "--compress_dims", type=list, default=None, required=True
        )
        vae_group.add_argument(
            "--decompress_dims", type=list, default=None, required=True
        )
        vae_group.add_argument("--lr", type=float, default=1e-3)
        vae_group.add_argument("--beta", type=float, default=1.0)
        return parser
