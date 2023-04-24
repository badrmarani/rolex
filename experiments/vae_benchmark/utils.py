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
import yaml

dirname = "experiments/vae_benchmark"
with open(os.path.join("training_config.yml"), "r") as stream:
    args = yaml.safe_load(stream)

training_base_config = BaseTrainerConfig(
    no_cuda=args["no_cuda"],
    num_epochs=args["n_epochs"],
    keep_best_on_train=args["keep_best_on_train"],
    output_dir=args["output_dir"],
    per_device_train_batch_size=args["train_batch_size"],
    per_device_eval_batch_size=args["train_batch_size"],
    optimizer_cls=args["optimizer_name"],
    learning_rate=args["lr"],
    optimizer_params=args["optimizer_params"],
    scheduler_cls=args["scheduler_name"],
    scheduler_params=args["scheduler_params"]
)

adversarial_trainer_config = AdversarialTrainerConfig(
    no_cuda=args["no_cuda"],
    num_epochs=args["n_epochs"],
    keep_best_on_train=args["keep_best_on_train"],
    output_dir=args["output_dir"],
    per_device_train_batch_size=args["train_batch_size"],
    per_device_eval_batch_size=args["train_batch_size"],
    autoencoder_optimizer_cls=args["optimizer_name"],
    autoencoder_learning_rate=args["lr"],
    autoencoder_optimizer_params=args["optimizer_params"],
    autoencoder_scheduler_cls=args["scheduler_name"],
    autoencoder_scheduler_params=args["scheduler_params"],
    discriminator_optimizer_cls=args["optimizer_name"],
    discriminator_learning_rate=args["lr"],
    discriminator_optimizer_params=args["optimizer_params"],
    discriminator_scheduler_cls=args["scheduler_name"],
    discriminator_scheduler_params=args["scheduler_params"]
)

coupled_optimizer_adversarial_trainer_config = CoupledOptimizerAdversarialTrainerConfig(
    no_cuda=args["no_cuda"],
    num_epochs=args["n_epochs"],
    keep_best_on_train=args["keep_best_on_train"],
    output_dir=args["output_dir"],
    per_device_train_batch_size=args["train_batch_size"],
    per_device_eval_batch_size=args["eval_batch_size"],
    encoder_optimizer_cls=args["optimizer_name"],
    encoder_learning_rate=args["lr"],
    encoder_optimizer_params=args["optimizer_params"],
    encoder_scheduler_cls=args["scheduler_name"],
    encoder_scheduler_params=args["scheduler_params"],
    decoder_optimizer_cls=args["optimizer_name"],
    decoder_learning_rate=args["lr"],
    decoder_optimizer_params=args["optimizer_params"],
    decoder_scheduler_cls=args["scheduler_name"],
    decoder_scheduler_params=args["scheduler_params"],
    discriminator_optimizer_cls=args["optimizer_name"],
    discriminator_learning_rate=args["lr"],
    discriminator_optimizer_params=args["optimizer_params"],
    discriminator_scheduler_cls=args["scheduler_name"],
    discriminator_scheduler_params=args["scheduler_params"]
)

coupled_optimizer_trainer_config = CoupledOptimizerTrainerConfig(
    no_cuda=args["no_cuda"],
    num_epochs=args["n_epochs"],
    keep_best_on_train=args["keep_best_on_train"],
    output_dir=args["output_dir"],
    per_device_train_batch_size=args["train_batch_size"],
    per_device_eval_batch_size=args["eval_batch_size"],
    encoder_optimizer_cls=args["optimizer_name"],
    encoder_learning_rate=args["lr"],
    encoder_optimizer_params=args["optimizer_params"],
    encoder_scheduler_cls=args["scheduler_name"],
    encoder_scheduler_params=args["scheduler_params"],
    decoder_optimizer_cls=args["optimizer_name"],
    decoder_learning_rate=args["lr"],
    decoder_optimizer_params=args["optimizer_params"],
    decoder_scheduler_cls=args["scheduler_name"],
    decoder_scheduler_params=args["scheduler_params"],
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
    fig.write_image(os.path.join(args["output_dir"], name + "_column_shapes.jpg"), width=800, height=800, scale=6)

    fig = quality_report.get_visualization(property_name="Column Pair Trends")
    fig.write_image(os.path.join(args["output_dir"], name + "_column_pair_trends.jpg"), width=800, height=800, scale=6)


    fig = utils.get_column_plot(
        real_data=dataset_x,
        synthetic_data=df_sampled_x,
        column_name="data_054",
        metadata=metadata.to_dict(),
    )
    fig.write_image(os.path.join(args["output_dir"], name + "_get_column_plot.jpg"), width=800, height=800, scale=6)

    quality_report.save(os.path.join(args["output_dir"], name + "quality_report.pkl"))
