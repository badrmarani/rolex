import pytorch_lightning as pl

import torch
from torch import nn, distributions
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
        
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.betas = betas

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=self.betas)
        return optimizer

    def _get_loss(self, batch):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        tot_loss, rec_loss, kld_loss = self._get_loss(batch)
        self.log("loss/fit/tot_loss", tot_loss, logger=True, on_epoch=True, on_step=False, prog_bar=True)
        self.log("loss/fit/kld_loss", kld_loss, logger=True, on_epoch=True, on_step=False, prog_bar=False)
        self.log("loss/fit/rec_loss", rec_loss, logger=True, on_epoch=True, on_step=False, prog_bar=False)
        return tot_loss


    def validation_step(self, batch, batch_idx):
        tot_loss, rec_loss, kld_loss = self._get_loss(batch)
        self.log("loss/val/tot_loss", tot_loss, logger=True, on_epoch=True, on_step=False, prog_bar=True)
        self.log("loss/val/kld_loss", kld_loss, logger=True, on_epoch=True, on_step=False, prog_bar=False)
        self.log("loss/val/rec_loss", rec_loss, logger=True, on_epoch=True, on_step=False, prog_bar=False)


class VAETrainer(MyBaseTrainer):

    def _get_loss(self, batch):
        x, y = batch
        xhat, mu, logvar = self.model(x)
        z = self.model.reparameterization(mu, logvar)
        return self.model.loss_function(x, xhat, mu, logvar)
