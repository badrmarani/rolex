import numpy as np
import pandas as pd
import torch
from sdmetrics.reports.single_table import QualityReport
from sdv.metadata import SingleTableMetadata
from tqdm import trange

from .utils import enable_dropout, lde, reproduce


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
@reproduce()
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
        mrange = trange(1, n_simulations + 1, desc=f"mutual_information")
    else:
        mrange = range(1, n_simulations + 1)
    for s in mrange:
        log_psm = []
        p_theta_0 = decoder(latent_sample)
        x_recon = p_theta_0.rsample()
        for m in range(n_sampled_outcomes):
            p_theta_m = decoder(latent_sample)
            log_psm += [p_theta_m.log_prob(x_recon).mean(-1)]
        log_psm = torch.stack(log_psm, dim=1)
        log_psm = torch.where(log_psm <= 0, log_psm, -log_psm)

        log_ps = -torch.tensor(
            n_sampled_outcomes, dtype=torch.float32
        ).log() + torch.logsumexp(log_psm, dim=1)

        log_hs_left = -torch.tensor(
            n_sampled_outcomes, dtype=torch.float32
        ).log() + torch.logsumexp(log_psm + torch.log(-log_psm), dim=1)

        log_hs_right = log_ps + torch.log(-log_ps)
        log_hs = lde(log_hs_left, log_hs_right)
        log_mi += [log_hs - log_ps]
    log_mi = torch.stack(log_mi, dim=1)
    log_mi_avg = -torch.tensor(
        n_simulations, dtype=torch.float32
    ).log() + torch.logsumexp(log_mi, dim=1)
    return log_mi_avg.exp()
