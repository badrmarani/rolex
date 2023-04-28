import torch
from torch import nn
from torch.utils import data

from uqvae.models import BaseVAE

import numpy as np
import random

from tqdm import tqdm
import os

def RegulWeights(model):
    regul = 0.0
    for layer in model.parameters():
        regul += layer.norm(2)
    return regul


class Trainer:
    def __init__(self, model: BaseVAE, logdir):
        self.model = model
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.generator = torch.Generator()
        self.generator.manual_seed(42)

        self.lambda_regul = 0.0
        self.logdir = logdir

    def worker_init_fn(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def set_optimizer(self, initial_lr:float=1e-5):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=initial_lr,
        )
    
    def set_scheduler(self, step=1, gamma=0.5):
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step, gamma)

    def fit(self, dataset:data.Dataset, fit_idx, val_idx, batch_size=32, epochs=30):
        self.model = self.model.to(self.device)

        fit_loader = data.DataLoader(
            dataset,
            batch_size,
            sampler=data.SubsetRandomSampler(fit_idx, self.generator),
            worker_init_fn=self.worker_init_fn,
            generator=self.generator,
            drop_last=True,
        )

        print(fit_loader.dataset)

        val_loader = data.DataLoader(
            dataset,
            batch_size,
            sampler=data.SubsetRandomSampler(val_idx, self.generator),
            worker_init_fn=self.worker_init_fn,
            generator=self.generator,
            drop_last=True,
        )

        best_loss = -1
        for epoch in range(1, epochs+1):
            print("\nepoch {}/{}".format(epoch, epochs))

            fit_batch_count = len(fit_idx)//batch_size
            fit_batch = tqdm(enumerate(fit_loader), total=fit_batch_count)

            total_loss, total_regul = 0.0, 0.0
            for i, batch in fit_batch:
                self.optimizer.zero_grad()
                loss, rec_loss, kld_loss = self.step(batch)

                regul = self.lambda_regul * RegulWeights(self.model) / fit_batch_count
                loss += regul

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()/fit_batch_count
                total_regul += regul

            with torch.no_grad():
                val_batch_count = len(val_idx)//batch_size
                val_batch = tqdm(enumerate(val_loader), total=val_batch_count)

                total_val_loss = 0.0
                for i, batch in val_batch:
                    loss, rec_loss, kld_loss = self.step(batch)
                    total_val_loss += loss.item()/val_batch_count

            if best_loss == -1 or total_val_loss < best_loss:
                best_loss = total_val_loss
                torch.save(
                    self.model.to("cpu").state_dict(),
                    os.path.join(
                        self.logdir, "{}_best_{}_epoch_{}_.pkl".format(
                            self.model.__class__.__name__,
                            best_loss,
                            epoch,
                        )
                    )
                )
            if self.scheduler:
                self.scheduler.step()


    def step(self, batch):
        x, y = (
            batch[0].to(self.device),
            batch[1].to(self.device),
        )


        print(x.size())
        recon_x, logsigma, mu, logvar = self.model(x)
        loss, rec_loss, kld_loss = self.model.loss_function(
            recon_x, x, logsigma, mu, logvar
        )

        return loss, rec_loss, kld_loss
