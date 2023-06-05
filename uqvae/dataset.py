import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils import data
from utils import reproduce


def split_dataset(df: pd.DataFrame):
    return (
        df.loc[:, "data_000":"data_135"],  # production params
        df.loc[:, "data_136":"data_196"],  # contextual params
        df.loc[:, "data_197":"data_211"],  # subs. quality properties
    )


def tensor_train_test_split(tensor: torch.Tensor, **kwargs):
    indx = torch.arange(0, tensor.size(0) - 1, 1)
    fit_indx, val_indx = train_test_split(indx, **kwargs)
    return tensor[fit_indx, :], tensor[val_indx, :]


def make_loader(dataset: np.array, seed: int = 42, shuffle: bool = False):
    if shuffle:
        np.random.shuffle(dataset)

    generator = torch.Generator()
    generator.manual_seed(seed)

    loader = data.DataLoader(
        data.TensorDataset(torch.from_numpy(dataset.astype("float32"))),
        batch_size=500,
        shuffle=False,
        worker_init_fn=reproduce,
        generator=generator,
    )

    return loader
