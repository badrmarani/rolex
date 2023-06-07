import argparse
import os
import shutil

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor

from uqvae import utils
from uqvae.dataset import JANUSDataModule
from uqvae.models.base import Decoder, Encoder
from uqvae.models.vae import BaseVAE


@utils.reproduce(42)
def main():
    parser = argparse.ArgumentParser()
    parser = JANUSDataModule.add_model_specific_args(parser)
    parser = BaseVAE.add_model_specific_args(parser)
    utils.add_default_trainer_args(parser)
    hparams = parser.parse_args()
    pl.seed_everything(hparams.seed)

    # create data
    datamodule = JANUSDataModule(hparams)

    # load model
    model = BaseVAE(hparams, data_dim=datamodule.data_dim)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="loss/val", save_top_k=1, save_last=True, mode="min"
    )

    if hparams.load_from_checkpoint is not None:
        print("Load from checkpoint")
    else:
        # Main trainer
        trainer = pl.Trainer(
            devices=1,
            accelerator="cuda" if hparams.cuda else "cpu",
            default_root_dir=hparams.root_dir,
            max_epochs=hparams.max_epochs,
            callbacks=[
                checkpoint_callback,
                LearningRateMonitor(logging_interval="epoch"),
            ],
            enable_progress_bar=True,
            num_sanity_val_steps=0,
        )

    trainer.fit(model, datamodule=datamodule)
    print(f"Training finished; end of script: rename {checkpoint_callback.best_model_path}")
    shutil.copyfile(
        checkpoint_callback.best_model_path,
        os.path.join(os.path.dirname(checkpoint_callback.best_model_path), "best.ckpt"),
    )

    print("Save the DataTransformer")
    torch.save(
        datamodule.data_transformer,
        os.path.join(os.path.dirname(checkpoint_callback.best_model_path), "data_transformer.ckpt"),
    )

if __name__ == "__main__":
    main()
