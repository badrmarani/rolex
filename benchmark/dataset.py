import pandas as pd

import torch
from torch.utils import data

from src.pythae.data.datasets import DatasetOutput

from sklearn.model_selection import train_test_split

from ctgan.data_transformer import DataTransformer

class PrepareJanusDataset(data.Dataset):
    def __init__(
            self,
            filename,
            n_classes_allowed,
            device,
            transform=False,
        ) -> None:
        super().__init__()
        self.filename = filename
        dataset, self.discrete_columns = self.prepare_janus_dataset(n_classes_allowed)
        self.n_samples, self.n_features = dataset.shape
        self.n_features -= 2
        
        self.x, self.y = (
            torch.from_numpy(dataset.iloc[:, :-2].values.astype("float32")).to(device),
            torch.from_numpy(dataset.iloc[:, -2:].values.astype("float32")).to(device),
        )

    def __getitem__(self, index):
        return (
            self.x[index],
            self.y[index],
        )

    def split(self, **kwargs):
        idx = torch.arange(0, self.n_samples, 1)
        train, test = train_test_split(idx, **kwargs)
        fit, val = train_test_split(train, **kwargs)
        return self[fit], self[val], self[test]

    def preprocess(self, n_classes_allowed):
        dataset = pd.read_csv(self.filename, sep=",")
        dataset.drop(columns=dataset.columns[:2], inplace=True, axis=1)
        dataset.drop(
            columns=["data_172", "data_173"],
            inplace=True,
            axis=1,
        )

        dataset = dataset.iloc[:, :-4]
        n_categorical_columns = {col: len(dataset[col].unique()) for col in dataset.columns}
        discrete_columns = [
            col
            for col in n_categorical_columns
            if n_categorical_columns[col] <= n_classes_allowed
        ]

        dataset = dataset.astype("float32")
        return dataset, discrete_columns

    def transform(self, data):
        transformer = DataTransformer()
        transformer.fit(data, self.discrete_columns)
        return transformer

    def get_metadata(self):
        pass

    def save_transformer(self, transformer):
        pass