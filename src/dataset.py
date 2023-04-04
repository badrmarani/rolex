import os
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import torch
from ctgan.data_transformer import DataTransformer
from sdv.metadata import SingleTableMetadata
from sdv.single_table.utils import detect_discrete_columns
from sklearn.model_selection import train_test_split
from torch.utils import data


def fit_val_split(
    tensor,
    train_size: float = 0.7,
    shuffle: bool = True,
):
    indx = torch.arange(0, tensor.size(0) - 1, 1)
    fit_indx, val_indx = train_test_split(indx, train_size=train_size, shuffle=shuffle)
    return (
        data.TensorDataset(tensor[fit_indx, :-2], tensor[fit_indx, -2:]),
        data.TensorDataset(tensor[val_indx, :-2], tensor[val_indx, -2:]),
    )


def prepare_quality_dataset(
    dataset: pd.DataFrame,
    targets: pd.DataFrame,
    train_size: float = 0.7,
    shuffle: bool = True,
    save_metadata: bool = None,
):
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(dataset)

    binary_columns = [col for col in dataset.columns if dataset[col].dtype == "int64"]

    for feat in binary_columns:
        metadata.update_column(
            column_name=feat,
            sdtype="categorical",
        )
    if save_metadata is not None:
        if os.path.exists(save_metadata):
            os.remove(save_metadata)
        metadata.save_to_json(filepath=save_metadata)

    discrete_columns = detect_discrete_columns(metadata, dataset)
    transformer = DataTransformer()
    transformer.fit(dataset, discrete_columns)

    transformed_data = transformer.transform(dataset)
    transformed_data = torch.from_numpy(transformed_data)

    targets = torch.from_numpy(targets.values.astype("float"))
    all_tensors = torch.cat((transformed_data, targets), dim=-1)
    fit, val = fit_val_split(all_tensors, train_size=train_size, shuffle=shuffle)

    return (
        transformer,
        fit,
        val,
    )
