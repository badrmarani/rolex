import pytorch_lightning as pl

import torch
from torch import nn
from torch.utils import data


class MyBaseTrainer(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        lr: float,
        betas,
        **kwargs,
    ):
        super(MyBaseTrainer, self).__init__()
        
        self.model = model(**kwargs)
        self.loss_fn = loss_fn
        self.lr = lr
        self.betas = betas

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=self.betas)
        return optimizer

    def _get_reconstruction_loss(self, batch):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.cache_loss = loss
        self.log("elbo_loss/fit", loss, logger=True, on_epoch=True, on_step=False, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("elbo_loss/val", loss, logger=True, on_epoch=True, on_step=False, prog_bar=True)


class VAETrainer(MyBaseTrainer):
    def _get_reconstruction_loss(self, batch):
        x, y = batch
        xhat, mu, logvar = self.model(x)
        return self.loss_fn(xhat, x, mu, logvar)

class CondVAETrainer(MyBaseTrainer):
    def _get_reconstruction_loss(self, batch):
        raise NotImplementedError
