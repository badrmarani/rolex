from argparse import ArgumentParser, Namespace
from typing import Any, List, Literal, Optional, Tuple, Union

import pandas as pd
import pytorch_lightning as pl
import torch
from ctgan.data_transformer import DataTransformer
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
from sdmetrics.single_column import (
    BoundaryAdherence,
    KSComplement,
    SingleColumnMetric,
    TVComplement,
)
from torch import nn

from ..processing.quality_metrics import compute_quality_scores, generate_fake_samples
from ..utils.parser import parse_list
from .base import BaseVAE


class TabularVAE(BaseVAE):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        data_transformer,
        lr: float,
        weight_decay: float,
        beta_on_kld: float,
        real_filtered_data: Optional[pd.DataFrame] = None,
        bayesian_decoder: str = None,
        **kwargs,
    ) -> None:
        super().__init__(
            encoder,
            decoder,
            lr,
            weight_decay,
            beta_on_kld,
            bayesian_decoder,
            **kwargs,
        )
        self.data_transformer = data_transformer
        self.real_filtered_data = real_filtered_data

        self.save_hyperparameters(
            ignore=[
                "encoder",
                "decoder",
                "data_transformer",
                "real_filtered_data",
                "data_transformer",
            ]
        )

    @torch.no_grad()
    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        """
        Calculate and log quality scores at the end of each training epoch.
        """
        self.decoder.eval()
        fake = generate_fake_samples(
            decoder=self.decoder,
            transformer=self.data_transformer.transformer,
            n_samples=self.real_filtered_data.shape[0],
        )
        scores = compute_quality_scores(
            self.real_filtered_data,
            fake,
            self.data_transformer.transformer,
            metrics=(KSComplement, TVComplement, BoundaryAdherence),
        )

        for k, v in scores.items():
            self.log(f"quality_score/{k}", v)

        k = "mean"
        v = (scores["TVComplement"] + scores["KSComplement"]) / 2
        self.log(f"quality_score/{k}", v, prog_bar=True)
        self.decoder.train()

    def reconstruction_loss_fn(
        self,
        x: torch.Tensor,
        recon_x: torch.Tensor,
        sigmas: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the reconstruction loss.

        Args:
            x (torch.Tensor): Original input data.
            recon_x (torch.Tensor): Reconstructed data mean.
            sigmas (torch.Tensor): Standard deviations of the reconstructed data.

        Returns:
            torch.Tensor: Reconstruction loss.

        """
        st = 0
        loss = []
        for column_info in self.data_transformer.transformer.output_info_list:
            for span_info in column_info:
                ed = st + span_info.dim
                if span_info.activation_fn != "softmax":
                    std = sigmas[st]
                    eq = x[:, st] - torch.tanh(recon_x[:, st])
                    loss.append((eq**2 / 2 / (std**2)).sum())
                    loss.append(torch.log(std) * x.size()[0])
                else:
                    ed = st + span_info.dim
                    loss.append(
                        nn.functional.cross_entropy(
                            recon_x[:, st:ed],
                            torch.argmax(x[:, st:ed], dim=-1),
                            reduction="sum",
                        )
                    )
                st = ed
        return sum(loss) / x.size(0)
