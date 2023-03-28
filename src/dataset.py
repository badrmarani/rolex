import torch
from torch.utils import data

import pandas as pd


class FullDS(data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        device: torch.device,
    ):
        super().__init__()

        self.x = torch.tensor(
            data=df[[col for col in df.columns if col.startswith("data")]].values,
            device=device,
            dtype=torch.float,
        )
        self.y = torch.tensor(
            df[["target_000", "target_001"]].values,
            device=device,
            dtype=torch.float,
        )

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, index):
        return (
            self.x[index, ...],
            self.y[index, ...],
        )
