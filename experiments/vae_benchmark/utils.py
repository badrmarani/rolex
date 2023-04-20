import torch
from torch import nn

from pythae.models import BaseAE
from pythae.models.nn import BaseEncoder, BaseDecoder
from pythae.models.base.base_utils import ModelOutput
from pythae.trainers.training_callbacks import MetricConsolePrinterCallback, TrainingCallback
from pythae.trainers.base_trainer.base_training_config import BaseTrainerConfig
from pythae.samplers import NormalSampler
from pythae.trainers import (
    BaseTrainerConfig,
    AdversarialTrainerConfig,
    CoupledOptimizerAdversarialTrainerConfig,
    CoupledOptimizerTrainerConfig,
)

from ray import air, tune

from sdmetrics.single_column import BoundaryAdherence
from sdmetrics.reports import utils
from sdmetrics.reports.single_table import QualityReport
from sdv.metadata import SingleTableMetadata

import pandas as pd
import os

output_dir = "experiments/vae_benchmark/all_models_logs"

NUM_EPOCHS = 100

training_base_config = BaseTrainerConfig(
    no_cuda=True,
    num_epochs=NUM_EPOCHS,
    keep_best_on_train=True,
    output_dir=output_dir,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    optimizer_cls="Adam",
    learning_rate=1e-4,
    optimizer_params={
        "betas": (0.91, 0.995),
        "weight_decay": 0.05,
    },
    scheduler_cls="ReduceLROnPlateau",
    scheduler_params={"patience": 5, "factor": 0.5}
)

adversarial_trainer_config = AdversarialTrainerConfig(
no_cuda=True,
    num_epochs=NUM_EPOCHS,
    keep_best_on_train=True,
    output_dir=output_dir,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    autoencoder_optimizer_cls="Adam",
    autoencoder_learning_rate=1e-4,
    autoencoder_optimizer_params={
        "betas": (0.91, 0.995),
        "weight_decay": 0.05,
    },
    autoencoder_scheduler_cls="ReduceLROnPlateau",
    autoencoder_scheduler_params={"patience": 5, "factor": 0.5},
    discriminator_optimizer_cls="Adam",
    discriminator_learning_rate=1e-4,
    discriminator_optimizer_params={
        "betas": (0.91, 0.995),
        "weight_decay": 0.05,
    },
    discriminator_scheduler_cls="ReduceLROnPlateau",
    discriminator_scheduler_params={"patience": 5, "factor": 0.5}
)

coupled_optimizer_adversarial_trainer_config = CoupledOptimizerAdversarialTrainerConfig(
    no_cuda=True,
    num_epochs=NUM_EPOCHS,
    keep_best_on_train=True,
    output_dir=output_dir,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    encoder_optimizer_cls="Adam",
    encoder_learning_rate=1e-4,
    encoder_optimizer_params={
        "betas": (0.91, 0.995),
        "weight_decay": 0.05,
    },
    encoder_scheduler_cls="ReduceLROnPlateau",
    encoder_scheduler_params={"patience": 5, "factor": 0.5},
    decoder_optimizer_cls="Adam",
    decoder_learning_rate=1e-4,
    decoder_optimizer_params={
        "betas": (0.91, 0.995),
        "weight_decay": 0.05,
    },
    decoder_scheduler_cls="ReduceLROnPlateau",
    decoder_scheduler_params={"patience": 5, "factor": 0.5},
    discriminator_optimizer_cls="Adam",
    discriminator_learning_rate=1e-4,
    discriminator_optimizer_params={
        "betas": (0.91, 0.995),
        "weight_decay": 0.05,
    },
    discriminator_scheduler_cls="ReduceLROnPlateau",
    discriminator_scheduler_params={"patience": 5, "factor": 0.5}
)

coupled_optimizer_trainer_config = CoupledOptimizerTrainerConfig(
    no_cuda=True,
    num_epochs=NUM_EPOCHS,
    keep_best_on_train=True,
    output_dir=output_dir,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    encoder_optimizer_cls="Adam",
    encoder_learning_rate=1e-4,
    encoder_optimizer_params={
        "betas": (0.91, 0.995),
        "weight_decay": 0.05,
    },
    encoder_scheduler_cls="ReduceLROnPlateau",
    encoder_scheduler_params={"patience": 5, "factor": 0.5},
    decoder_optimizer_cls="Adam",
    decoder_learning_rate=1e-4,
    decoder_optimizer_params={
        "betas": (0.91, 0.995),
        "weight_decay": 0.05,
    },
    decoder_scheduler_cls="ReduceLROnPlateau",
    decoder_scheduler_params={"patience": 5, "factor": 0.5},
)

CONFIGS = {
    "adversarial_trainer_config": adversarial_trainer_config, 
    "coupled_optimizer_adversarial_trainer_config": coupled_optimizer_adversarial_trainer_config, 
    "coupled_optimizer_trainer_config": coupled_optimizer_trainer_config, 
    "training_base_config": training_base_config, 
}


class TBLogger(MetricConsolePrinterCallback):
    def __init__(self, writer):
        super().__init__()
        self.writer = writer
        self.current_indx = 0

    def on_log(self, training_config: BaseTrainerConfig, logs, **kwargs):
        epoch_train_loss = logs.get("train_epoch_loss", None)
        epoch_eval_loss = logs.get("eval_epoch_loss", None)

        self.current_indx += 1
        self.writer.add_scalar("loss/train", epoch_train_loss, self.current_indx)
        self.writer.add_scalar("loss/eval", epoch_eval_loss, self.current_indx)


class RayLogger(TrainingCallback):
    def __init__(self):
        super().__init__()

    def on_epoch_end(self, training_config: BaseTrainerConfig, **kwargs):
        metrics = kwargs.pop("metrics")
        tune.report(eval_epoch_loss=metrics["eval_epoch_loss"])
class Encoder(BaseEncoder):
    def __init__(self, inp_size, emb_sizes, lat_size, mtype="base"):
        BaseEncoder.__init__(self)

        seq = []
        for i, emb_size in enumerate(emb_sizes):
            if not i:
                seq += [
                    nn.Linear(inp_size, emb_size), nn.ReLU(),
                ]
            else:
                seq += [
                    nn.Linear(pre_emb_size, emb_size), nn.ReLU(),
                ]
            pre_emb_size = emb_size
        

        self.seq = nn.Sequential(*seq)

        self.embedding = nn.Linear(pre_emb_size, lat_size)

        self.mtype = mtype
        if self.mtype.lower() == "svae":
            tout = 1
        else:
            tout = lat_size
        
        self.log_covariance = nn.Linear(pre_emb_size, tout)
        
    def forward(self, x: torch.Tensor) -> ModelOutput:
        output = ModelOutput()

        emb = self.seq(x)

        output["embedding"] = self.embedding(emb)
        if self.mtype.lower() == "svae":            
            output["log_concentration"] = self.log_covariance(emb)
        else:
            output["log_covariance"] = self.log_covariance(emb)
        return output


class Decoder(BaseDecoder):
    def __init__(self, lat_size, emb_sizes, out_size):
        BaseDecoder.__init__(self)

        emb_sizes = emb_sizes[::-1]

        seq = []
        for i, emb_size in enumerate(emb_sizes):
            if not i:
                seq += [
                    nn.Linear(lat_size, emb_size), nn.ReLU(),
                ]
            else:
                seq += [nn.Linear(pre_emb_size, emb_size)]
                if i != len(emb_sizes)-1:
                    seq += [nn.ReLU()]

            pre_emb_size = emb_size
        
        self.reconstruction = nn.Sequential(*seq)
        
        
    def forward(self, x: torch.Tensor) -> ModelOutput:
        output = ModelOutput()

        output["reconstruction"] = self.reconstruction(x)
        return output

def save_sdmetrics(model: BaseAE, dataset_x: pd.DataFrame, name):
    normal_sampler = NormalSampler(model)    
    sampled_x = normal_sampler.sample(num_samples=dataset_x.shape[0])

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(dataset_x)

    quality_report = QualityReport()
    df_sampled_x = pd.DataFrame(
        sampled_x.detach().cpu().numpy(),
        columns=dataset_x.columns,
        index=dataset_x.index
    )

    quality_report.generate(dataset_x, df_sampled_x, metadata.to_dict())
    
    fig = quality_report.get_visualization(property_name="Column Shapes")
    fig.write_image(os.path.join(output_dir, name + "_column_shapes.jpg"), width=800, height=800, scale=6)

    fig = quality_report.get_visualization(property_name="Column Pair Trends")
    fig.write_image(os.path.join(output_dir, name + "_column_pair_trends.jpg"), width=800, height=800, scale=6)


    fig = utils.get_column_plot(
        real_data=dataset_x,
        synthetic_data=df_sampled_x,
        column_name="data_054",
        metadata=metadata.to_dict(),
    )
    fig.write_image(os.path.join(output_dir, name + "_get_column_plot.jpg"), width=800, height=800, scale=6)

    quality_report.save(os.path.join(output_dir, name + "quality_report.pkl"))
