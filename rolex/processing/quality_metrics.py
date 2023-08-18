from typing import Dict, Optional, Tuple

import pandas as pd
import torch
from ctgan.data_transformer import DataTransformer
from sdmetrics.single_column import SingleColumnMetric
from torch import nn

from ..metrics.utils import enable_dropout


@torch.no_grad()
def generate_fake_samples(
    decoder: nn.Module,
    transformer: DataTransformer,
    n_samples: int,
    n_steps: Optional[int] = 10,
) -> pd.DataFrame:
    """
    Generate fake samples using the provided decoder and transformer.

    Args:
        decoder (nn.Module): The decoder module to generate samples.
        transformer (DataTransformer): The transformer for data conversion.
        n_samples (int): Number of fake samples to generate.
        n_steps (Optional[int]): Number of steps for generating samples.

    Returns:
        pd.DataFrame: A DataFrame containing the generated fake samples.
    """
    decoder.eval()
    enable_dropout(decoder)
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
    """
    Calculate the score for a given metric using real and fake data.

    Args:
        real_df (pd.DataFrame): Real data for the metric computation.
        fake_df (pd.DataFrame): Fake data for the metric computation.
        metric (SingleColumnMetric): The metric to compute the score for.

    Returns:
        float: The computed metric score.
    """
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
    """
    Compute quality scores for a set of metrics using real and fake data.

    Args:
        real_df (pd.DataFrame): Real data for quality score computation.
        fake_df (pd.DataFrame): Fake data for quality score computation.
        transformer (DataTransformer): The transformer for data conversion.
        metrics (Tuple[SingleColumnMetric, ...]): Tuple of metrics to compute.

    Returns:
        Dict[str, float]: A dictionary of metric names and their corresponding scores.
    """
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
