import torch
from torch import nn
from torch.utils import data
from torch.utils import tensorboard

from uqvae.models import BaseVAE

import numpy as np
import copy
import random

from tqdm import tqdm
import os

from hydra.utils import get_class


import hydra

def RegulWeights(model):
    regul = 0.0
    for layer in model.parameters():
        regul += layer.norm(2)
    return regul

class Trainer:
    def __init__(
        self,
        logdir,
        device,
        dataset,
        fit_idx,
        val_idx,
        batch_size
    ):
        self.device = device

        self.generator = torch.Generator()
        self.generator.manual_seed(42)

        self.logdir = logdir

        self.batch_size = batch_size
        self.fit_idx = fit_idx
        self.val_idx = val_idx
        self.fit_loader = data.DataLoader(
            dataset,
            batch_size,
            sampler=data.SubsetRandomSampler(fit_idx, self.generator),
            worker_init_fn=self.worker_init_fn,
            generator=self.generator,
            drop_last=True,
        )

        self.val_loader = data.DataLoader(
            dataset,
            batch_size,
            sampler=data.SubsetRandomSampler(val_idx, self.generator),
            worker_init_fn=self.worker_init_fn,
            generator=self.generator,
            drop_last=True,
        )


    def worker_init_fn(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def set_optimizer(self, **kwargs):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            **kwargs,
        )
    
    def set_scheduler(self, step=1, gamma=1):
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step, gamma)

    def construct_model(
        self,
        model_target,
        encoder,
        decoder,
        model_config,
        categorical_columns,
    ):
        self.model_config = model_config
        self.categorical_columns = categorical_columns
        self.model_name = model_target.split(".")[-1]
        self.model = get_class(model_target)(
            encoder=encoder,
            decoder=decoder,
            model_config=model_config,
            categorical_columns=categorical_columns,            
        )


    def fit(self, epochs=30):
        print(f"\t+ Moving Module to device: {self.device.type.upper()}")
        self.model = self.model.to(self.device)

        model_save_dir = os.path.join(self.logdir, self.model_name)
        self.writer = tensorboard.SummaryWriter(model_save_dir)

        best_loss = -1
        for epoch in range(1, epochs+1):
            print("\n{} \tEpoch {}/{}".format(
                self.model_name,
                epoch,
                epochs
            ))

            fit_batch_count = len(self.fit_idx)//self.batch_size
            fit_batch = tqdm(enumerate(self.fit_loader), total=fit_batch_count)

            total_loss, total_regul = 0.0, 0.0
            for i, batch in fit_batch:
                self.optimizer.zero_grad()
                loss, rec_loss, kld_loss = self.step(batch, epoch)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()/fit_batch_count
                fit_batch.set_postfix({"loss": "{:.4f}".format(total_loss)}, refresh=True)
                fit_batch.set_description(f"fit")

            with torch.no_grad():
                val_batch_count = len(self.val_idx)//self.batch_size
                val_batch = tqdm(enumerate(self.val_loader), total=val_batch_count)
                val_batch.set_description(f"val")

                total_val_loss = 0.0
                for i, batch in val_batch:
                    loss, rec_loss, kld_loss = self.step(batch, epoch)
                    total_val_loss += loss.item()/val_batch_count
                    val_batch.set_postfix({"loss": "{:.4f}".format(total_val_loss)}, refresh=True)

            if best_loss == -1 or total_val_loss < best_loss:
                best_loss = total_val_loss
                torch.save(
                    {
                        "model": copy.deepcopy(self.model).to("cpu").state_dict(),
                        "model_config": self.model_config,
                        "categorical_columns": self.categorical_columns,
                    },
                    os.path.join(
                        model_save_dir, "best_{}.pkl".format(
                            self.model_name,
                        )
                    )
                )

            if self.scheduler:
                self.scheduler.step()

            self.writer.add_scalar("loss/fit", total_loss, epoch)
            self.writer.add_scalar("loss/val", total_val_loss, epoch)
        self.writer.close()
        print("\t+ Training finished")

    def step(self, batch, epoch):
        x, y = (
            batch[0].to(self.device),
            batch[1].to(self.device),
        )

        recon_x, mu, logvar = self.model(x)
        loss, rec_loss, kld_loss = self.model.get_loss(
            recon_x, x, mu, logvar, epoch
        )


        return loss, rec_loss, kld_loss
