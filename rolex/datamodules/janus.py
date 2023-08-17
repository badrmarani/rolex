import warnings
from pathlib import Path
from typing import List, Tuple

import joblib
import pandas as pd

from ..processing.mode_normalization import ModeNormalization
from .tabular import TabularDataModule

warnings.filterwarnings("ignore")


def load_janus(filename: str) -> Tuple[pd.DataFrame, ...]:
    dataset = pd.read_csv(filename, sep=",", index_col=0)
    dataset.drop(columns=["ROW_ID"], inplace=True)
    dataset = split_dataset(dataset)
    dataset[1].drop(columns=["data_172", "data_173"], inplace=True)
    return dataset


def split_dataset(temp_data: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
    return (
        temp_data.loc[:, "data_000":"data_135"],
        temp_data.loc[:, "data_136":"data_196"],
        temp_data.loc[:, "data_197":"data_211"],
        temp_data.loc[:, "target_000":"target_001"],
    )


class JanusDataModule(TabularDataModule):
    def __init__(
        self,
        data: pd.DataFrame,
        batch_size: int,
        val_split: float = 0,
        num_workers: int = 1,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        correlation_threshold: float = 0.8,
        oversample_quantile: float = 0.5,
        qq_threshold: float = 0.96,
        categorical_columns: List[str] | None = None,
        n_samples_to_transform: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            data,
            batch_size,
            val_split,
            num_workers,
            pin_memory,
            persistent_workers,
            correlation_threshold,
            oversample_quantile,
            qq_threshold,
            categorical_columns,
            n_samples_to_transform,
            **kwargs,
        )

    def prepare(self) -> None:
        path = Path("data/janus_msn.ckpt").absolute()
        if not path.exists():
            self.msn = ModeNormalization(
                correlation_threshold=self.correlation_threshold,
                oversample_quantile=self.oversample_quantile,
                qq_threshold=self.qq_threshold,
                categorical_columns=self.categorical_columns,
                n_samples_to_transform=self.n_samples_to_transform,
            )

            self.new_data = self.msn.fit_transform(self.real_data)
            joblib.dump(self.msn, path, compress=3)
        else:
            print("found!")
            self.msn = joblib.load(path)
            self.new_data = self.msn.transform(self.real_data)

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
