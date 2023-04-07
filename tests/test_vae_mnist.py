import warnings

import pytorch_lightning as pl
import torch
from lightning.pytorch import loggers
from torch import nn
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import MNIST

from src.losses import ELBOLoss
from src.models import VAE
from src.training import VAETrainer

warnings.filterwarnings("ignore")

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        # transforms.Normalize((0.0,), (1.0,)),
        transforms.ConvertImageDtype(torch.float64),
        transforms.Lambda(torch.flatten),
    ]
)

fit_loader = data.DataLoader(
    MNIST("tests/mnist/", train=True, download=True, transform=transform),
    batch_size=200,
    # num_workers=2,
)

val_loader = data.DataLoader(
    MNIST("tests/mnist/", train=True, download=True, transform=transform),
    batch_size=200,
    # num_workers=2,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_fn = ELBOLoss(device=device)

net = VAE(
    dropout=False,
    inp_size=28 * 28,
    emb_size=(28 * 28) // 2,
    lat_size=2,
).to(device)

vae_trainer = VAETrainer(
    model=net,
    loss_fn=loss_fn,
    lr=1e-3,
    betas=(0, 0.99),
)

logger = loggers.TensorBoardLogger(save_dir="tests/mnist_test_logs/")
trainer = pl.Trainer(max_epochs=10, logger=logger, accelerator="auto")
trainer.fit(vae_trainer, fit_loader, val_loader)
