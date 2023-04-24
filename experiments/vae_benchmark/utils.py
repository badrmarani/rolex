import torch
from torch import nn

import numpy as np

from pythae.data.datasets import BaseDataset
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

import warnings
warnings.filterwarnings("ignore")

from ray import air, tune

from sdmetrics.reports.single_table import QualityReport
from sdv.metadata import SingleTableMetadata

from sklearn.model_selection import train_test_split

import pandas as pd
import os
import yaml
import datetime

training_signature = str(datetime.datetime.now())[0:19].replace(" ", "_").replace(":", "-")
dirname = "experiments/vae_benchmark"

with open(os.path.join(dirname, "training_config.yml"), "r") as stream:
    args = yaml.load(stream, Loader=yaml.FullLoader)


def get_configs(model_name:str):
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

    if model_name.lower() in ["adversarial_ae", "factorvae"]:
        training_config = adversarial_trainer_config
    elif model_name.lower() in ["vaegan"]:
        training_config = coupled_optimizer_adversarial_trainer_config
    elif model_name.lower() in ["piwae"]:
        training_config = coupled_optimizer_trainer_config
    else:
        training_config = training_base_config
    return training_config

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

def save_quality_report(model: BaseAE, dataset: pd.DataFrame, name):
    normal_sampler = NormalSampler(model)
    fake_dataset = normal_sampler.sample(num_samples=dataset.shape[0])

    binary_columns = [col for col in dataset.columns if dataset[col].dtype == "int64"]
    dataset[binary_columns] = dataset[binary_columns].astype("category")

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(dataset)

    quality_report = QualityReport()
    df_fake_dataset = pd.DataFrame(
        fake_dataset.detach().cpu().numpy(),
        columns=dataset.columns,
        index=dataset.index
    )

    quality_report.generate(dataset, df_fake_dataset, metadata.to_dict())
    quality_report.save(name)

def prepare_dataset(filename: str):
    data = pd.read_csv(filename, sep=",")
    data.reset_index()
    data = data.iloc[:, 2:-6]
    
    # remove columns with nan values
    tmp = data.isna().any()
    na_columns = tmp[lambda x: x].index.to_list()
    data.drop(columns=na_columns, axis=1, inplace=True)
    return data

def get_dataset(filename: str, **kwargs):
    data = prepare_dataset(filename)
    n_samples = data.shape[0]
    idx = np.arange(0, n_samples, 1)
    train_idx, eval_idx = train_test_split(idx, **kwargs)
    train_idx, test_idx = train_test_split(train_idx, **kwargs)
    train = torch.from_numpy(data.iloc[train_idx,:].values).to(torch.float)
    test = torch.from_numpy(data.iloc[test_idx,:].values).to(torch.float)
    eval = torch.from_numpy(data.iloc[eval_idx,:].values).to(torch.float)

    return {
        "train": {
            "pandas": data.iloc[train_idx,:],
            "torch": BaseDataset(train, torch.ones(n_samples,)),
        },
        "test": {
            "pandas": data.iloc[test_idx,:],
            "torch": BaseDataset(test, torch.ones(n_samples,)),
        },
        "eval": {
            "pandas": data.iloc[eval_idx,:],
            "torch": BaseDataset(eval, torch.ones(n_samples,)),
        },
    }


def get_stats_per_model(model: BaseAE, dataset: pd.DataFrame, name):
    normal_sampler = NormalSampler(model)
    fake_dataset = normal_sampler.sample(num_samples=dataset.shape[0])

    nll = model.get_nll(model, dataset, n_samples=3)


from pythae.models import AutoModel
from pathlib import Path
import numpy as np

def get_stats(dirname: str, dataset, repeat_id):
    all_model_dir = [x for x in os.listdir(dirname) if not x.startswith("tensorboard")]
    for model_dir in all_model_dir:
        model_name = model_dir.split("_training")[0]
        path = Path(
            dirname,
            model_dir,
            model_name + f"_rep_{repeat_id}",
            "final_model/",
        )

        trained_model = AutoModel.load_from_folder(path)
        quality_report_save_path = os.path.join(
            dirname,
            model_dir,
            model_name + f"_rep_{repeat_id}",
            model_name+"_quality_report.pkl",
        )
        save_quality_report(trained_model, dataset, quality_report_save_path)

# dirname = "experiments/vae_benchmark/all_models_logs"
# dataset = get_dataset("data/fulldataset.csv", test_size=0.10, random_state=42)
# eval = dataset["eval"]["pandas"]
# get_stats(dirname, eval, 1)
