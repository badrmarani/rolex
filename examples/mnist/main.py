import os
from datetime import datetime

import hydra
import numpy as np
import torch
from dataset import MNISTDataModule
from omegaconf import open_dict
from torch import nn

from rolex.dataset import DataWeighter
from rolex.models.base import Decoder, Encoder, MLPRegressor
from rolex.models.vae import VAE
from rolex.trainer import Trainer

@hydra.main(config_path=".", config_name="config", version_base=None)
def main(config):
    dtype = getattr(torch, config.dtype)
    torch.set_default_dtype(dtype)
    with open_dict(config):
        config.log_dir = (
            "./logs/"
            + datetime.now()
            .strftime(f"{config.experiment_name}_%B_%d_%Y_%H_%M_%S")
            .lower()
        )
        config.log_dir = os.path.join(config.root, config.log_dir)

    # model definition
    regressor = MLPRegressor(config.embedding_dim)
    encoder = Encoder(
        config.data_dim,
        config.compress_dims,
        config.embedding_dim,
        config.bayesian,
    )
    decoder = Decoder(
        config.embedding_dim,
        config.decompress_dims,
        config.data_dim,
        p=config.p,
        bayesian=config.bayesian,
    )
    model = VAE(encoder, decoder, regressor=regressor)

    # data definition
    data_weighter = DataWeighter(config)
    data_module = MNISTDataModule(config, data_weighter=data_weighter, valid=True)
    data_module.setup(root=config.root)

    train_loader = data_module.train_dataloader()
    valid_loader = data_module.valid_dataloader()

    # init trainer
    trainer = Trainer(model=model)
    trainer.fit(
        train_loader=train_loader,
        valid_loader=valid_loader,
        config=config,
        dtype=dtype,
    )

    # optimize
    trainer.optimize(regressor=model.regressor)


if __name__ == "__main__":
    main()
