from typing import Dict, Optional, Tuple

import pandas as pd
import torch
from ctgan.data_transformer import DataTransformer
from sdmetrics.single_column import SingleColumnMetric
from torch import nn


@torch.no_grad()
def generate_fake_samples(
    decoder: nn.Module,
    transformer: DataTransformer,
    n_samples: int,
    n_steps: Optional[int] = 10,
) -> pd.DataFrame:
    decoder.train()
    t = next(decoder.parameters())
    noise = torch.randn(
        size=(n_samples, decoder.embedding_dim),
        device=t.device,
        dtype=t.dtype,
    )
    step = noise.size(0) // n_steps
    for i in range(0, noise.size(0), step):
        ns = noise[i : i + step, ...]
        recon_x, recon_std = decoder(ns)
        recon_x = torch.tanh(recon_x)
        f = transformer.inverse_transform(
            recon_x.cpu().numpy(), recon_std.cpu().numpy()
        ).astype(t.dtype.__str__().split(".")[-1])
        if not i:
            fake = f
        else:
            fake = pd.concat((fake, f), ignore_index=True)
    fake.reset_index(drop=True, inplace=True)
    return fake


def sdmetrics_wrapper(
    real_df: pd.DataFrame,
    fake_df: pd.DataFrame,
    metric: SingleColumnMetric,
) -> float:
    score = 0.0
    for column in real_df.columns:
        score += metric.compute(
            real_data=real_df.loc[:, column],
            synthetic_data=fake_df.loc[:, column],
        )
    score /= len(real_df.columns)
    return score


def compute_quality_scores(
    real_df: pd.DataFrame,
    fake_df: pd.DataFrame,
    transformer: DataTransformer,
    metrics: Tuple[SingleColumnMetric, ...],
) -> Dict[str, float]:
    cat_columns, con_columns = [], []
    for col in transformer._column_transform_info_list:
        if col.column_type == "discrete":
            cat_columns += [col.column_name]
        else:
            con_columns += [col.column_name]

    dtype = "float64"
    real_df.reset_index(drop=True, inplace=True)
    fake_df.reset_index(drop=True, inplace=True)

    scores = dict()
    for metric in metrics:
        name = metric.__name__
        if name == "TVComplement" and len(cat_columns):
            scores[name] = sdmetrics_wrapper(
                real_df=real_df.loc[:, cat_columns].astype("category"),
                fake_df=fake_df.loc[:, cat_columns].astype("category"),
                metric=metric,
            )
        else:
            scores[name] = sdmetrics_wrapper(
                real_df=real_df.loc[:, con_columns].astype(dtype),
                fake_df=fake_df.loc[:, con_columns].astype(dtype),
                metric=metric,
            )
    return scores
