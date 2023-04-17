import os
import sys

my_full_path = os.path.abspath(".")
sys.path.insert(1, my_full_path)

import pandas as pd
import pytorch_lightning as pl
import torch
from lightning.pytorch import loggers as pl_loggers
from torch import nn
from torch.utils import data

from src.dataset import clean_dataset, fit_val_split
from src.models import VAE
from src.training import VAETrainer

from ctgan.data_transformer import DataTransformer

# DATASET PREP.

df = pd.read_csv("data/fulldataset.csv", sep=",")
df = clean_dataset(df)
all_x, all_y = df.iloc[:, :-2], df.iloc[:, -2:]

transformer = DataTransformer()
discrete_columns = [col for col in all_x.columns if all_x[col].dtype == "int64"]
transformer.fit(all_x, discrete_columns)
transformed_x = transformer.transform(all_x)

new_all_x = torch.from_numpy(transformed_x.astype("float"))
new_all_y = torch.from_numpy(all_y.values.astype("float"))
all_tensors = torch.cat((new_all_x, new_all_y), dim=-1)
fit, val = fit_val_split(all_tensors, train_size=0.7, shuffle=True)

dropout = False
inp_size = fit.tensors[0].size(-1)
emb_size = inp_size // 2
lat_size = 8

lr = 1e-3
betas = (0.5, 0.999)
kl_factor = 1.0,

batch_size = 32
fit_loader = data.DataLoader(fit, batch_size=batch_size)
val_loader = data.DataLoader(val, batch_size=batch_size)

# MODEL

# def loss_function(xhat, x, mu, logvar, factor = 1e-5):
#     rec_loss = nn.functional.mse_loss(xhat, x, reduction="mean")
#     kld_loss = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
#     return rec_loss + kld_loss * factor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from src.losses import ELBOLoss
loss_function = ELBOLoss(device)

inp_size = transformer.output_dimensions
net = VAE(
    dropout=False,
    inp_size=inp_size,
    emb_size=inp_size//2,
    lat_size=2,
).to(device)

my_trainer = VAETrainer(
    model = net,
    loss_fn = loss_function,
    lr = lr,
    betas = betas,
)

logger = pl_loggers.TensorBoardLogger(save_dir="final_log")
trainer = pl.Trainer(max_epochs=5000, logger=logger, accelerator="auto")
trainer.fit(my_trainer, fit_loader, val_loader)
