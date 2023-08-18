import warnings
from pathlib import Path
from typing import List, Tuple

import joblib
import pandas as pd

from ..processing.mode_normalization import ModeNormalization
from .tabular import TabularDataModule

warnings.filterwarnings("ignore")


def load_janus(filename: str) -> Tuple[pd.DataFrame, ...]:
    """
    Load Janus dataset from a CSV file and split it into four segments :
        - production parameters
        - contextual parameters
        - substitute quality properties
        - and target quality properties

    Args:
        filename (str): Path to the CSV file.

    Returns:
        Tuple[pd.DataFrame, ...]: Tuple containing different segments of the dataset.
    """
    dataset = pd.read_csv(filename, sep=",", index_col=0)
    dataset.drop(columns=["ROW_ID"], inplace=True)
    dataset = split_dataset(dataset)
    dataset[1].drop(columns=["data_172", "data_173"], inplace=True)
    return dataset


def split_dataset(data: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
    """
    Split the dataset into four segments.
        - production parameters
        - contextual parameters
        - substitute quality properties
        - and target quality properties

    Args:
        data (pd.DataFrame): Input dataset.

    Returns:
        Tuple[pd.DataFrame, ...]: Tuple containing different segments of the dataset.
    """
    return (
        data.loc[:, "data_000":"data_135"],
        data.loc[:, "data_136":"data_196"],
        data.loc[:, "data_197":"data_211"],
        data.loc[:, "target_000":"target_001"],
    )


class JanusDataModule(TabularDataModule):
    """
    Data module for the Janus dataset.

    Args:
        data (pd.DataFrame): Input data.
        batch_size (int): Batch size.
        val_split (float): Validation split ratio (default: 0).
        num_workers (int): Number of workers for data loading (default: 1).
        pin_memory (bool): Whether to use pinned memory for data loading (default: True).
        persistent_workers (bool): Whether to keep workers alive between epochs (default: True).
        correlation_threshold (float): Threshold for correlation filtering (default: 0.8).
        oversample_quantile (float): Quantile for oversampling (default: 0.5).
        qq_threshold (float): Threshold for quantile-quantile filtering (default: 0.96).
        categorical_columns (List[str] | None): List of categorical column names (default: None).
        n_samples_to_transform (float | None): Number of samples to use for normalization (default: None).
        **kwargs: Additional keyword arguments.

    Attributes:
        msn (ModeNormalization): ModeNormalization instance for data preprocessing.
        new_data (torch.Tensor): Preprocessed data.
        data_dim (int): Dimension of the data.

    """

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
        """
        Prepare the dataset for training.

        Creates or loads a ModeNormalization instance for preprocessing the data.
        """
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
