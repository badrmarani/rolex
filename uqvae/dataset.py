import argparse
import warnings

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from ctgan.data_transformer import DataTransformer
from sklearn.model_selection import train_test_split
from torch.utils import data
from torchvision import datasets

from .utils import reproduce

warnings.filterwarnings("ignore")

NUM_WORKERS = 0


@torch.no_grad()
def generate_synthetic_data(model, n_samples, embedding_dim, data_transformer, device):
    noise = torch.randn(n_samples, embedding_dim, device=device, dtype=torch.float32)
    pxz = model.decoder(noise)
    sigmas = model.decoder.logvar.mul(0.5).exp().cpu().numpy()
    fake = pxz.loc.cpu().numpy()
    fake = data_transformer.inverse_transform(fake, sigmas=sigmas)
    return fake


def split_dataset(df: pd.DataFrame):
    return (
        df.loc[:, "data_000":"data_135"],  # production params
        df.loc[:, "data_136":"data_196"],  # contextual params
        df.loc[:, "data_197":"data_211"],  # subs. quality properties
        df.loc[:, "target_000":"target_001"],
    )


def tensor_train_test_split(tensor: pd.DataFrame, **kwargs):
    indx = torch.arange(0, tensor.shape[0] - 1, 1)
    fit_indx, val_indx = train_test_split(indx, **kwargs)
    return fit_indx, val_indx


def make_loader(dataset: np.array, seed: int = 42, shuffle: bool = False):
    if shuffle:
        np.random.shuffle(dataset)

    generator = torch.Generator()
    generator.manual_seed(seed)

    loader = data.DataLoader(
        data.TensorDataset(torch.from_numpy(dataset.astype("float32"))),
        batch_size=500,
        shuffle=False,
        worker_init_fn=reproduce,
        generator=generator,
    )

    return loader


def load(filename: str):
    dataset = pd.read_csv(filename, sep=",", index_col=0)
    dataset.drop(columns=["ROW_ID"], inplace=True)
    dataset = split_dataset(dataset)
    dataset[1].drop(columns=["data_172", "data_173"], inplace=True)
    return dataset


class JANUSDataModule(pl.LightningDataModule):
    def __init__(self, hparams) -> None:
        super().__init__()

        self.dataset_path = hparams.dataset_path
        self.test_size = hparams.test_size
        self.batch_size = hparams.batch_size
        self.transform_data = hparams.transform_data

        self.specific_setup()

    @staticmethod
    def add_model_specific_args(parent_parser):
        data_group = parent_parser.add_argument_group(title="data")
        data_group.add_argument(
            "--dataset_path", type=str, required=True, help="path to csv file"
        )
        data_group.add_argument("--batch_size", type=int, default=500)
        data_group.add_argument(
            "--test_size",
            type=float,
            default=0.05,
            help="Fraction of val data.",
        )
        data_group.add_argument("--transform_data", default=False, action="store_true")
        return parent_parser

    def production_dataset_preprocess(self):
        if self.transform_data:
            print("Transform data")
            self.data_transformer = DataTransformer()
            self.data_transformer.fit(self.data_train)
            self.data_train = torch.from_numpy(
                self.data_transformer.transform(self.data_train).astype("float32")
            )

            self.data_val = torch.from_numpy(
                self.data_transformer.transform(self.data_val).astype("float32")
            )
        else:
            print("No data tranformation")
            self.data_train = torch.from_numpy(self.data_train.values.astype("float32"))

            self.data_val = torch.from_numpy(self.data_val.values.astype("float32"))

        self.data_dim = self.data_train.size(-1)

    def setup(self, stage: str) -> None:
        pass

    def specific_setup(self):
        data, _, _, _ = load(self.dataset_path)
        indx_train, indx_val = tensor_train_test_split(
            data,
            test_size=self.test_size,
            random_state=42,
            shuffle=False,
        )

        self.data_train = data.iloc[indx_train]
        self.data_val = data.iloc[indx_val]

        self.production_dataset_preprocess()
        self.make_tensor_dataset()

    def make_tensor_dataset(self):
        self.train_dataset = data.TensorDataset(self.data_train)
        self.val_dataset = data.TensorDataset(self.data_val)

    def train_dataloader(self):
        return data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=NUM_WORKERS,
            drop_last=True,
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=NUM_WORKERS,
            drop_last=True,
        )
