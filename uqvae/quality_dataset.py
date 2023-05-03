import pandas as pd

import torch
from torch.utils import data

from ctgan.data_transformer import DataTransformer

def preprocess_janus_dataset(filename, n_classes_allowed=10):
    dataset = pd.read_csv(filename, index_col=0, sep=",")
    dataset.drop(columns=dataset.columns[0], inplace=True, axis=1)
    dataset.drop(
        columns=["data_172", "data_173"],
        inplace=True,
        axis=1,
    )

    dataset = dataset.iloc[:, :-4]
    if n_classes_allowed > 2:
        n_unique_vals_per_column = {
            col: len(dataset[col].unique())
            for col in dataset.columns
        }

        discrete_columns = [
            col
            for col in n_unique_vals_per_column
            if n_unique_vals_per_column[col] <= n_classes_allowed
        ]
    else:
        discrete_columns = [
            col
            for col in dataset.columns
            if dataset[col].dtype == "int64"
        ]

    # dataset = dataset.astype("float32")
    return dataset, discrete_columns


class JanusDataset(data.Dataset):
    def __init__(
        self,
        dataset,
        discrete_columns=None,
        device=None,
        preprocess_dataset=False,
    ):

        self.device = device
        self.dataset = dataset
        _, self.n_features = self.dataset.shape
        self.n_features -= 2

        df_x = self.dataset.iloc[:, :-2]
        if preprocess_dataset:
            print("\t+ Transforming Janus data...", end=" ")
            self.transformer = DataTransformer()
            print("Fit...", end=" ")
            tmp = df_x.sample(50, axis=0)
            self.transformer.fit(tmp, discrete_columns)
            print("Transform...", end=" ")
            x = self.transformer.transform(tmp).astype("float32")
            print("Done.")
            self.x = torch.from_numpy(x).to(self.device)
            self.n_features = self.transformer.output_dimensions

        else:
            self.x = torch.from_numpy(df_x.values.astype("float32")).to(self.device)
        self.y = torch.from_numpy(self.dataset.iloc[:, -2:].values.astype("float32")).to(self.device)
        self.n_samples = self.x.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return (
            self.x[index],
            self.y[index],
        )
