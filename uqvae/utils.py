import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils import data

from .losses import ELBOLoss
import pandas as pd




def train_one_epoch(
    epoch,
    model,
    loader,
    loss_fn,
    optimizer,
    device,
    log_intervall=50
):
    model.train()
    train_loss = 0.0
    train_rec_loss, train_kld_loss = 0.0, 0.0
    for i, batch in enumerate(loader, start=1):
        x, _ = batch
        x = x.to(device)
        optimizer.zero_grad()
        xhat, mu, logvar = model(x)

        loss, rec_loss, kld_loss = loss_fn(x, xhat, mu, logvar)
        train_loss += loss.item()
        train_rec_loss += rec_loss.item()
        train_kld_loss += kld_loss.item()
        loss.backward()
        optimizer.step()

        if not i % log_intervall:
            print(
                "train epoch: {} [{}/{} ({:.0f}%)] elbo_loss: {:.6f} kld_loss {:.6f} rec_loss {:.6f}".format(
                    epoch,
                    i * len(batch),
                    len(batch) * len(loader),
                    100.0 * i / len(loader),
                    loss.item() / len(batch),
                    kld_loss.item() / len(batch),
                    rec_loss.item() / len(batch),
                )
            )

    print(
        "epoch: {} avg_elbo_loss: {:.6f} avg_kld_loss {:.6f} avg_rec_loss {:.6f}".format(
            epoch,
            train_loss / len(loader.dataset),
            train_kld_loss / len(loader.dataset),
            train_rec_loss / len(loader.dataset),
        )
    )

    return model
