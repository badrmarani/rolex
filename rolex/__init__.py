import random
from argparse import ArgumentParser, Namespace
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger


def cli(
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    root: Union[str, Path],
    exp_name: str,
    args: Namespace,
    version: Optional[str] = None,
) -> None:
    """
    Command-line interface for training a model.

    Args:
        model (pl.LightningModule): The PyTorch Lightning model to be trained.
        datamodule (pl.LightningDataModule): The PyTorch Lightning data module.
        root (Union[str, Path]): Root directory for logging and saving checkpoints.
        exp_name (str): Experiment name for logging purposes.
        args (Namespace): Parsed command-line arguments.
        version (Optional[str], optional): Version identifier for logging. Defaults to None.
    """
    if isinstance(root, str):
        root = Path(root)

    monitor = "quality_score/mean"
    mode = "max"

    if isinstance(args.seed, int):
        pl.seed_everything(seed=args.seed, workers=True)

    # logger
    tb_logger = TensorBoardLogger(
        str(root / "logs"),
        name=exp_name,
        default_hp_metric=False,
        log_graph=args.log_graph,
        version=version,
    )

    # callbacks
    save_checkpoints = ModelCheckpoint(
        monitor=monitor,
        mode=mode,
        save_last=True,
        save_weights_only=True,
    )
    # Select the best model, monitor the lr and stop if NaN
    callbacks = [
        save_checkpoints,
        LearningRateMonitor(logging_interval="step"),
        EarlyStopping(
            monitor=monitor, patience=np.inf, check_finite=True, mode=mode
        ),
    ]

    # trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=callbacks,
        logger=tb_logger,
        deterministic=(args.seed is not None),
    )

    trainer.fit(model, datamodule)


def init_args(
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
) -> Namespace:
    """
    Initialize command-line arguments for training.

    Args:
        model (pl.LightningModule): The PyTorch Lightning model.
        datamodule (pl.LightningDataModule): The PyTorch Lightning data module.

    Returns:
        Namespace: Parsed command-line arguments.
    """
    parser = ArgumentParser("rolex")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--test", type=int, default=None)
    parser.add_argument("--log_graph", dest="log_graph", action="store_true")

    parser = pl.Trainer.add_argparse_args(parser)
    if model is not None:
        parser = model.add_model_specific_args(parser)

    if datamodule is not None:
        parser = datamodule.add_argparse_args(parser)

    return parser.parse_args()


@contextmanager
def seed_everything(seed=42):
    """
    Seed random number generators for reproducibility.

    Args:
        seed (int, optional): Seed value for random number generators. Defaults to 42.
    """
    random.seed(seed)
    np.random.default_rng(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True
    yield
