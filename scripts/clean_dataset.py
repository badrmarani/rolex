from sdv.metadata import SingleTableMetadata

import numpy as np
import pandas as pd
import os

def split_dataset(df: pd.DataFrame):
    return (
        df.loc[:, "data_000":"data_135"],  # production params
        df.loc[:, "data_136":"data_196"],  # contextual params
        df.loc[:, "data_197":"data_211"],  # subs. quality properties
    )


CORRELATION_THRESH = 0.9

filename = "data/fulldataset.csv"
os.makedirs("data/", exist_ok=True)

df = pd.read_csv(filename, sep=",")

df = df.iloc[:, :-4]
df = df.drop(columns=df.columns[:2], axis=1, inplace=False)

tmp = df.iloc[:, :-2].isna().any()
na_columns = tmp[lambda x: x].index.to_list()

df = df.drop(columns=na_columns, inplace=False)
binary_columns = [col for col in df.columns if df[col].dtype == "int64"]

not_na_targets_indxs = df[["target_000", "target_001"]].notna().all(1)

dataset = df[not_na_targets_indxs]

one_value_cols = pd.Series(
    {
        c: len(dataset[c].unique())
        for c in dataset
        if len(dataset[c].unique()) == 1
    }
).index.to_list()

binary_columns = list(set(binary_columns) - set(one_value_cols))
dataset = dataset.drop(columns=one_value_cols, axis=1, inplace=False)
dataset_x = dataset.iloc[:, :-2]
dataset_y = dataset.iloc[:, -2:]

num_dataset_x = dataset_x[dataset_x.columns.difference(binary_columns)]
corr_num_dataset_x = num_dataset_x.corr().abs()

corr_num_dataset_x.loc[:,:] = np.tril(corr_num_dataset_x.values, k=-1)
corr_num_dataset_x_pairs = corr_num_dataset_x.unstack().sort_values(ascending=False)
corr_num_dataset_x_pairs

perf_corr_dataset_pairs = corr_num_dataset_x_pairs[corr_num_dataset_x_pairs >= CORRELATION_THRESH].index.to_list()
dataset_x = dataset_x.drop(
    columns=[x[0] for x in perf_corr_dataset_pairs],
    axis=1,
    inplace=False,
)

dataset_x[binary_columns] = dataset_x[binary_columns].astype("category")

dataset_x1, dataset_x2, dataset_x3 = split_dataset(dataset_x)

dataset_y.to_csv("data/full_no_nans_dataset_y.csv", sep=",", index=False)
dataset_x.to_csv("data/full_no_nans_dataset_x.csv", sep=",", index=False)
dataset_x1.to_csv("data/full_no_nans_dataset_x1.csv", sep=",", index=False)
dataset_x2.to_csv("data/full_no_nans_dataset_x2.csv", sep=",", index=False)
dataset_x3.to_csv("data/full_no_nans_dataset_x3.csv", sep=",", index=False)

dataset_x12 = pd.concat((dataset_x1, dataset_x2), axis=1)
dataset_x12.to_csv("data/full_no_nans_dataset_x12.csv", sep=",", index=False)
