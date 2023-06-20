import numpy as np
import pandas as pd
import torch
from pytorch_metric_learning import distances
from sdmetrics.reports.single_table import QualityReport
from sdv.metadata import SingleTableMetadata
from torch import nn, distributions
from tqdm import trange

from .utils import enable_dropout, lde, reproduce


def denormalize_targets(y, scaler_path: str, return_scaler: bool = False):
    scaler = pd.read_pickle(scaler_path)
    scaler_info = pd.DataFrame(
        data=np.vstack(
            (
                scaler.mean_.reshape(1, -1),
                scaler.scale_.reshape(1, -1),
            )
        ).reshape(2, -1),
        columns=scaler.feature_names_in_,
        index=["mean", "scale"],
    )
    scaler_info = torch.from_numpy(scaler_info.values.astype("float32"))
    y = y * scaler_info[1, :] + scaler_info[0, :]
    if return_scaler:
        return y, scaler
    else:
        return y


def emp_boundary_adherence(fake, scaler_path: str):
    scaler = pd.read_pickle(scaler_path)
    scaler_info = pd.DataFrame(
        data=np.vstack(
            (
                scaler.mean_.reshape(1, -1),
                scaler.scale_.reshape(1, -1),
            )
        ).reshape(2, -1),
        columns=scaler.feature_names_in_,
        index=["mean", "scale"],
    )

    scaler_info = scaler_info[
        [
            "ZGFM_1400_TEMPSCP1",
            "ZGFM_1403_TEMPSCP2",
            "ZGFM_1406_TEMPSCP3",
            "ZGFM_1420_TEMPSMP1",
            "ZGFM_1423_TEMPSMP2",
            "ZGFM_1426_TEMPSMP3",
        ]
    ]
    scaler_info = torch.from_numpy(scaler_info.values.astype("float32"))

    fake = torch.from_numpy(
        fake[
            ["data_034", "data_037", "data_040", "data_043", "data_046", "data_049"]
        ].values.astype("float32")
    )
    fake = fake * scaler_info[1, :] + scaler_info[0, :]

    valid1 = torch.where(
        torch.logical_and(fake[:, 0] <= fake[:, 1], fake[:, 1] <= fake[:, 2]), 1.0, 0.0
    )
    valid2 = torch.where(
        torch.logical_and(fake[:, 3] <= fake[:, 4], fake[:, 4] <= fake[:, 5]), 1.0, 0.0
    )
    return (valid1 + valid2) / 2


def boundary_adherence(
    fake_tensor: torch.Tensor,
    real_production_dataset: pd.DataFrame,
):
    real_production_dataset = torch.from_numpy(
        real_production_dataset.describe().T[["min", "max"]].T.values.astype("float32")
    )
    valid = torch.where(
        torch.logical_and(
            real_production_dataset[0, :] <= fake_tensor,
            fake_tensor <= real_production_dataset[1, :],
        ),
        1.0,
        0.0,
    )
    return valid.mean(dim=-1)


def make_quality_report(real, fake):
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(real)

    report = QualityReport()
    report.generate(real.astype("float32"), fake.astype("float32"), metadata.to_dict())

    return report, metadata


def sdmetrics_wrapper(real, fake, metric):
    score = 0.0
    for column in real.columns:
        score += metric.compute(real_data=real[column], synthetic_data=fake[column])
    score /= len(real.columns)
    return score


@torch.no_grad()
def mutual_information(
    decoder,
    latent_sample,
    n_simulations,
    n_sampled_outcomes,
    verbose=True,
):
    decoder.eval()
    enable_dropout(decoder)
    log_mi = []
    if verbose:
        looper = trange(1, n_simulations + 1, desc=f"mutual_information")
    else:
        looper = range(1, n_simulations + 1)
    for _ in looper:
        log_psm = []
        p_theta_0 = decoder(latent_sample)
        x_recon = p_theta_0.rsample()
        for _ in range(n_sampled_outcomes):
            p_theta_m = decoder(latent_sample)
            log_psm += [p_theta_m.log_prob(x_recon).mean(-1)]
        log_psm = torch.stack(log_psm, dim=1)
        log_psm = torch.where(log_psm <= 0, log_psm, -log_psm)

        log_ps = (
            - torch.tensor(n_sampled_outcomes, dtype=torch.float32).log()
            + torch.logsumexp(log_psm, dim=1)
        )

        log_hs_left = (
            - torch.tensor(n_sampled_outcomes, dtype=torch.float32).log()
            + torch.logsumexp(log_psm + torch.log(-log_psm), dim=1)
        )

        log_hs_right = log_ps + torch.log(-log_ps)
        log_hs = lde(log_hs_left, log_hs_right)
        log_mi += [log_hs - log_ps]
    log_mi = torch.stack(log_mi, dim=1)
    log_mi_avg = (
        - torch.tensor(n_simulations, dtype=torch.float32).log()
        + torch.logsumexp(log_mi, dim=1)
    )
    return log_mi_avg.exp()


@torch.no_grad()
def effective_sample_size(
    decoder,
    latent_sample,
    n_simulations,
    n_sampled_outcomes,
    reduction="none",
    verbose=True
):
    """
    Ref: https://arxiv.org/abs/1912.05651v3.
    """
    decoder.eval()
    enable_dropout(decoder)
    log_ess_x = []
    looper = trange if verbose else range
    for _ in looper(n_simulations):
        log_px = []
        p0 = decoder(latent_sample)
        x = p0.rsample()
        for _ in range(n_sampled_outcomes):
            pi = decoder(latent_sample)
            log_px += [pi.log_prob(x).sum(-1)]
        log_px = torch.stack(log_px, dim=1)
        log_wx = log_px - log_px.logsumexp(dim=1)[:, None]
        log_ess_x += [-torch.logsumexp(2 * log_wx, dim=1)]
    log_ess_x = torch.stack(log_ess_x, dim=1)
    log_ess_z = -torch.tensor(n_simulations).log() + \
        log_ess_x.logsumexp(dim=-1)
    if reduction.lower() == "none":
        return log_ess_z
    elif reduction.lower() == "mean":
        return log_ess_z.mean(-1)
    else:
        print(f"{reduction.lower()} is not implemented.")

@torch.no_grad()
def marginal_log_likelihood_estimator(
    model,
    data,
    n_sampled_outcomes=1000,
    reduction="none",
    verbose=True
):
    """
    Estimate the marginal log-likelihood using importance sampling method.
    Ref: Appendix D of the paper, https://arxiv.org/abs/1312.6114.
    """
    model.eval()
    enable_dropout(model)
    log_p = []
    prior = distributions.Normal(0.0, 1.0)
    looper = trange if verbose else range
    for _ in looper(n_sampled_outcomes):
        qzx, pxz = model(data)
        z = qzx.rsample()
        log_p += [
            qzx.log_prob(z).sum(-1)
            - prior.log_prob(z).sum(-1)
            - pxz.log_prob(data).sum(-1)
        ]
    log_p = torch.stack(log_p, dim=-1)
    log_p = torch.tensor(data.size(0)).log() - log_p.logsumexp(1)
    if reduction.lower() == "none":
        return log_p
    elif reduction.lower() == "mean":
        return log_p.mean(-1)
    else:
        print(f"{reduction.lower()} is not implemented.")
