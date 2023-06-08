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
GENERATOR = torch.Generator()
GENERATOR.manual_seed(42)


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


def tensor_train_test_split_indx(tensor: pd.DataFrame, **kwargs):
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
    def __init__(self, hparams, data_weighter) -> None:
        super().__init__()

        self.save_hyperparameters(hparams)
        self.dataset_path = hparams.dataset_path
        self.test_size = hparams.test_size
        self.batch_size = hparams.batch_size
        self.transform_data = hparams.transform_data
        self.data_weighter = data_weighter

        self.specific_setup()
        self.production_dataset_preprocess()
        if self.hparams.semi_supervised_learning:
            self.set_weights()
        self.make_tensor_dataset()

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
        data_group.add_argument(
            "--semi_supervised_learning", default=False, action="store_true"
        )
        return parent_parser

    def production_dataset_preprocess(self):
        if self.transform_data:
            print("Transform data")
            self.data_transformer = DataTransformer()
            self.data_transformer.fit(self.data_train, ())
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

    def setup(self, stage: str):
        pass

    def set_weights(self, data, properties):
        weights = DataWeighter.uniform_weights(properties.values.astype("float32"))
        self.sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
        )

    def specific_setup(self):
        if not self.hparams.semi_supervised_learning:
            data, _, _, _ = load(self.dataset_path)
            indx_train, indx_val = tensor_train_test_split_indx(
                data,
                test_size=self.test_size,
                random_state=42,
                shuffle=False,
            )
            self.data_train = data.iloc[indx_train]
            self.data_val = data.iloc[indx_val]
        else:
            data, _, _, target = load(self.dataset_path)
            mask = np.where(target.notna().all(1))[0]
            data = data.iloc[mask, :]
            # right now, we're focusing on the first quality target.
            target = target.iloc[mask, 0]
            indx_train, indx_val = tensor_train_test_split_indx(
                data,
                test_size=self.test_size,
                random_state=42,
                shuffle=False,
            )
            self.data_train = data.iloc[indx_train]
            self.target_train = torch.from_numpy(
                target.iloc[indx_train].values.astype("float32")
            )
            self.data_val = data.iloc[indx_val]
            self.target_val = torch.from_numpy(
                target.iloc[indx_val].values.astype("float32")
            )

    def make_tensor_dataset(self):
        self.train_dataset = data.TensorDataset(self.data_train)
        self.val_dataset = data.TensorDataset(self.data_val)
        if self.hparams.semi_supervised_learning:
            self.train_dataset = data.TensorDataset(self.data_train, self.target_train)
            self.val_dataset = data.TensorDataset(self.data_val, self.target_val)

    def train_dataloader(self):
        return data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=NUM_WORKERS,
            shuffle=False,
            worker_init_fn=reproduce,
            generator=GENERATOR,
            sampler=(
                self.train_sampler if self.hparams.semi_supervised_learning else None
            ),
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=NUM_WORKERS,
            shuffle=False,
            worker_init_fn=reproduce,
            generator=GENERATOR,
            sampler=(
                self.val_sampler if self.hparams.semi_supervised_learning else None
            ),
        )

    def set_weights(self):
        weights_train = self.data_weighter.weighting_function(self.target_train)
        weights_val = self.data_weighter.weighting_function(self.target_val)

        self.train_sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights_train,
            num_samples=len(weights_train),
            replacement=True,
        )
        self.val_sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights_val,
            num_samples=len(weights_val),
            replacement=True,
        )


class DataWeighter:
    weight_types = ["uniform", "property_values", "rank"]

    def __init__(self, hparams):
        if hparams.weight_type in ["uniform"]:
            self.weighting_function = DataWeighter.uniform_weights
        elif hparams.weight_type in ["property_values"]:
            self.weighting_function = DataWeighter.property_values_weights
        else:
            raise NotImplementedError

        self.weight_type = hparams.weight_type

    @staticmethod
    def normalize_weights(weights: np.array):
        return weights / np.mean(weights)

    @staticmethod
    def uniform_weights(properties: torch.Tensor):
        return torch.ones_like(properties)

    @staticmethod
    def property_values_weights(properties: torch.Tensor):
        return properties

    @staticmethod
    def add_weight_args(parser: argparse.ArgumentParser):
        weight_group = parser.add_argument_group("weighting")
        weight_group.add_argument(
            "--weight_type",
            type=str,
            default="uniform",
            choices=DataWeighter.weight_types,
            required=False,
        )
        weight_group.add_argument(
            "--rank_weight_k",
            type=float,
            default=None,
            help="k parameter for rank weighting",
        )
        return parser
