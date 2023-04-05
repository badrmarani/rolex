import os
import sys

my_full_path = os.path.abspath(".")
sys.path.insert(1, my_full_path)

import pandas as pd

import torch
from torch import nn
from torch.utils import data

from ctgan import TVAE

import pytorch_lightning as pl
from lightning.pytorch import loggers as pl_loggers

from src.models import VAE
from src.training import VAETrainer
from src.dataset import clean_dataset, fit_val_split

# DATASET PREP.

df = pd.read_csv("data/fulldataset.csv", sep=",")
df = clean_dataset(df)
all_x, all_y = df.iloc[:, :-2], df.iloc[:, -2:]

new_all_x = torch.from_numpy(all_x.values.astype("float"))
new_all_y = torch.from_numpy(all_y.values.astype("float"))

all_tensors = torch.cat((new_all_x, new_all_y), dim=-1)
fit, val = fit_val_split(all_tensors, train_size=0.7, shuffle=True)


dropout = True
inp_size = fit.tensors[0].size(-1)
emb_size = inp_size // 2
lat_size = 2

lr = 1e-3
betas = (0.5, 0.999)

batch_size = 32
fit_loader = data.DataLoader(fit, batch_size=batch_size)
val_loader = data.DataLoader(val, batch_size=batch_size)

# MODEL

def loss_function(xhat, x, mu, logvar, factor = 1e-5):
    rec_loss = nn.functional.mse_loss(xhat, x, reduction="mean")
    kld_loss = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
    return rec_loss + kld_loss * factor

logger = pl_loggers.TensorBoardLogger(save_dir="final_log")

trainer = pl.Trainer(
    max_epochs = 5000,
    logger = logger,
    accelerator = "auto",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

my_trainer = VAETrainer(
    model = VAE,
    loss_fn = loss_function,
    lr = lr,
    betas = betas,
    dropout = dropout,
    inp_size = inp_size,
    emb_size = emb_size,
    lat_size = lat_size,
)


trainer.fit(my_trainer, fit_loader, val_loader)

