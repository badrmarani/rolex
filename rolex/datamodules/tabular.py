import warnings
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Optional, Tuple, Union

import joblib
import pandas as pd
import pytorch_lightning as pl
from torch import utils

from ..processing.mode_normalization import ModeNormalization
from ..utils.parser import parse_list

warnings.filterwarnings("ignore")


class TabularDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data: pd.DataFrame,
        batch_size: int,
        val_split: float = 0.0,
        num_workers: int = 1,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        correlation_threshold: float = 0.80,
        oversample_quantile: float = 0.5,
        qq_threshold: float = 0.96,
        categorical_columns: Optional[List[str]] = None,
        n_samples_to_transform: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers

        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        self.correlation_threshold = correlation_threshold
        self.oversample_quantile = oversample_quantile
        self.qq_threshold = qq_threshold
        self.categorical_columns = categorical_columns
        self.n_samples_to_transform = n_samples_to_transform

        self.real_data = data.copy()
        self.real_data.astype("float64")
        self.prepare()

    @classmethod
    def add_argparse_args(
        cls,
        parent_parser: ArgumentParser,
    ) -> ArgumentParser:
        p = parent_parser.add_argument_group("datamodule")
        p.register("type", list, parse_list)
        p.add_argument("--version", type=str, choices=["tabular", "image"])
        p.add_argument("--root", type=str, default="./data/")
        p.add_argument("--batch_size", type=int, default=512)
        p.add_argument("--val_split", type=float, default=0)
        p.add_argument("--num_workers", type=int, default=4)
        p.add_argument("--correlation_threshold", type=float, default=0.8)
        p.add_argument("--oversample_quantile", type=float, default=0.5)
        p.add_argument("--qq_threshold", type=float, default=0.96)
        p.add_argument("--categorical_columns", type=list, default=None)
        p.add_argument("--n_samples_to_transform", type=float, default=None)
        return parent_parser

    def _data_loader(
        self,
        dataset: utils.data.Dataset,
        shuffle: bool = False,
    ) -> utils.data.DataLoader:
        return utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def prepare(self) -> None:
        self.msn = ModeNormalization(
            correlation_threshold=self.correlation_threshold,
            oversample_quantile=self.oversample_quantile,
            qq_threshold=self.qq_threshold,
            categorical_columns=self.categorical_columns,
            n_samples_to_transform=self.n_samples_to_transform,
        )

        self.new_data = self.msn.fit_transform(self.real_data)
        self.data_dim = self.new_data.size(1)
        a, b = [], []
        for col in self.msn.transformer._column_transform_info_list:
            if col.column_type == "discrete":
                a += [col.column_name]
            else:
                b += [col.column_name]
        print("Found {} discrete features.".format(len(a)))
        print("Found {} continuous features.".format(len(b)))
        print("Size of the training data:", self.new_data.size(0))
        print("Number of features:", self.new_data.size(1))

        self.real_filtered_data = self.real_data.loc[:, a + b].astype("float64")
        print(self.real_filtered_data.shape)
        del a, b

    def setup(self, stage: Optional[str] = None) -> None:
        self.new_data = utils.data.TensorDataset((self.new_data))
        self.train_data, self.val_data = utils.data.random_split(
            self.new_data, [0.8, 0.2]
        )

    def train_dataloader(self) -> utils.data.DataLoader:
        return self._data_loader(self.train_data, shuffle=True)

    def val_dataloader(self) -> utils.data.DataLoader:
        return self._data_loader(self.val_data)
