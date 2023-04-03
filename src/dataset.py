import os

import numpy as np
import pandas as pd
import torch
from ctgan.data_transformer import DataTransformer
from sdv.metadata import SingleTableMetadata
from sdv.single_table.utils import detect_discrete_columns
from sklearn.model_selection import train_test_split
from torch.utils import data


def cvt_int64_to_bool(df: pd.DataFrame):
    categorical_features = [col for col in df.columns if df[col].dtype == "int64"]
    df[categorical_features] = df[categorical_features].astype("bool")
    return df


def get_metadata(
    df: pd.DataFrame,
    save_metadata: str = None,
):
    df = cvt_int64_to_bool(df)
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)

    if save_metadata is not None:
        if os.path.exists(save_metadata):
            os.remove(save_metadata)
        metadata.save_to_json(filepath=save_metadata)

    return metadata


def fit_data_transformer(
    dataset: pd.DataFrame,
    metadata,
):
    transformer = DataTransformer()
    discrete_columns = detect_discrete_columns(metadata, dataset)
    transformer.fit(dataset, discrete_columns)
    return transformer


class PrepareQualityDataset():
    def __init__(
        self,
        df: pd.DataFrame,
        train_size: float = 0.7,
        shuffle: bool = True,
    ) -> None:
        super(PrepareQualityDataset, self).__init__()

        self.df = cvt_int64_to_bool(df)
        self.train_size = train_size
        self.shuffle = shuffle


    def fit_val_split(
        self,
        train_size: float = 0.7,
        shuffle: bool = True,
    ):
        indx = torch.arange(0, self.df.shape[0] - 1, 1)
        fit_indx, val_indx = train_test_split(
            indx, train_size=train_size, shuffle=shuffle
        )
        return (
            self.df.iloc[fit_indx, :],
            self.df.iloc[val_indx, :],
        )


    @property
    def run(self):
        fit, val = self.fit_val_split(self.train_size, self.shuffle)
        metadata = get_metadata(fit)

        transformer = fit_data_transformer(fit, metadata)
        fit_trs = transformer.transform(fit)
        val_trs = transformer.transform(val)

        self.fit = data.TensorDataset(torch.from_numpy(fit_trs.astype(np.float64)))
        self.val = data.TensorDataset(torch.from_numpy(val_trs.astype(np.float64)))

        return (
            self.fit, self.val
        )
