import torch
from sdmetrics.reports.single_table import QualityReport
from sdv.metadata import SingleTableMetadata
from tqdm import trange

from .utils import reproduce, lde, enable_dropout


@torch.no_grad()
def generate_synthetic_data(model, n_samples, embedding_dim, data_transformer, device):
    noise = torch.randn(n_samples, embedding_dim, device=device, dtype=torch.float32)
    pxz = model.decoder(noise)
    sigmas = model.decoder.logvar.mul(0.5).exp().cpu().numpy()
    fake = pxz.loc.cpu().numpy()
    fake = data_transformer.inverse_transform(fake, sigmas=sigmas)
    return fake


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
