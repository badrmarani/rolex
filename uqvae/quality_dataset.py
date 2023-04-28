import pandas as pd

import torch
from torch.utils import data

def prepare_dataset(filename, n_classes_allowed=10, preprocess_dataset=False):
    dataset = pd.read_csv(filename, index_col=0, sep=",")
    dataset.drop(columns=dataset.columns[0], inplace=True, axis=1)
    dataset = dataset.iloc[:, :-4]

    n_unique_vals_per_column = {
        col: len(dataset[col].unique())
        for col in dataset.columns
    }

    if preprocess_dataset:
        discrete_columns = []
        for col in n_unique_vals_per_column:
            if n_unique_vals_per_column[col] >= n_classes_allowed:
                dataset[col] = dataset[col].astype("float32")
            else:
                discrete_columns.append(col)
                dataset[col] = dataset[col].astype("category")
            return dataset, discrete_columns
    else:
        dataset = dataset.astype("float32")
        return dataset


class JanusDataset(data.Dataset):
    def __init__(self, dataset, n_classes_allowed=10, device=None, preprocess_dataset=False):
        # self.filename = filename
        self.n_classes_allowed = n_classes_allowed
        self.preprocess_dataset = preprocess_dataset
        
        if device is None:
            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        else:
            self.device = device
        
        # self.dataset = pd.read_csv(self.filename, sep=",", index_col=0)
        
        self.dataset = dataset
        self.n_samples, self.n_features = self.dataset.shape
        self.x, self.y = self.prepare_dataset()

    def prepare_dataset(self):
        # self.dataset.drop(columns=self.dataset.columns[0], inplace=True, axis=1)
        # self.dataset = self.dataset.iloc[:, :-4]

        # n_unique_vals_per_column = {
        #     col: len(self.dataset[col].unique())
        #     for col in self.dataset.columns
        # }

        # if self.preprocess_dataset:
        #     self.discrete_columns = []
        #     for col in n_unique_vals_per_column:
        #         if n_unique_vals_per_column[col] >= self.n_classes_allowed:
        #             self.dataset[col] = self.dataset[col].astype("float32")
        #         else:
        #             self.discrete_columns.append(col)
        #             self.dataset[col] = self.dataset[col].astype("category")
        # else:
        #     self.dataset = self.dataset.astype("float32")

        
        return (
            torch.from_numpy(self.dataset.iloc[:, :-2].values).to(self.device),
            torch.from_numpy(self.dataset.iloc[:, -2:].values).to(self.device),
        )

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return (
            self.x[index, ...],
            self.y[index],
        )
