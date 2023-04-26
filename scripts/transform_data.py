import torch

from ctgan.data_transformer import DataTransformer
from sdv.metadata import SingleTableMetadata
from sdv.single_table.utils import detect_discrete_columns

import os
import pandas as pd

from sklearn.model_selection import train_test_split


def get_dataset(filename: str):
    data = pd.read_csv(filename, sep=",")
    data.reset_index()
    data = data.iloc[:, 2:-6]

    # remove columns with nan values
    tmp = data.isna().any()
    na_columns = tmp[lambda x: x].index.to_list()
    print("columns with nan values", na_columns)
    data.drop(columns=na_columns, axis=1, inplace=True)
    return data


filename = os.path.normpath("data/fulldataset.csv")
data = get_dataset(filename)

idx = list(range(data.shape[0]))
train_idx, eval_idx = train_test_split(idx, test_size=0.10, random_state=42)
train_idx, test_idx = train_test_split(train_idx, test_size=0.10, random_state=42)

train = data.iloc[train_idx, :]
test = data.iloc[test_idx, :]
eval = data.iloc[eval_idx, :]

binary_columns = [col for col in data.columns if data[col].dtype == "int64"]
data[binary_columns] = data[binary_columns].astype("category")

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data)

discrete_columns = detect_discrete_columns(metadata, data)
data_transformer = DataTransformer()
data_transformer.fit(train, discrete_columns)

transformed_train = data_transformer.transform(train)
transformed_test = data_transformer.transform(test)
transformed_eval = data_transformer.transform(eval)

savepath = os.path.normpath("data/transformed_data_tvae")
os.makedirs(savepath, exist_ok=True)
torch.save(
    {
        "train": (train, transformed_train),
        "test": (test, transformed_test),
        "eval": (eval, transformed_eval),
        "data_transformer": data_transformer,
    },
    os.path.join(savepath, "transformed_data.pkl"),
)
