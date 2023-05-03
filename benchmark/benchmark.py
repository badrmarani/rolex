import torch
import pickle

import importlib

from types import SimpleNamespace

from uqvae.quality_dataset import JanusDataset, preprocess_janus_dataset
from uqvae.trainer import Trainer
from uqvae.models import VAE, Encoder, Decoder

import hydra
from hydra.utils import get_class

import os
import yaml

from sklearn.model_selection import train_test_split

import omegaconf

import gc

import warnings

warnings.filterwarnings("ignore")


class Benchmark:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(self.config.device)

        self.compress_dims = config.compress_dims
        self.decompress_dims = config.decompress_dims
        self.embedding_dim = config.embedding_dim

        print("\t+ Preprocessing Janus data")
        if (
            config.dataset.preprocessed_dataset_path is not None
            and config.dataset.preprocess_dataset
        ):
            self.dataset, self.discrete_columns = torch.load(
                self.config.dataset.preprocessed_dataset_path,
                map_location=self.device,
            )
        else:
            dataset, self.discrete_columns = preprocess_janus_dataset(
                config.dataset.filename, config.dataset.n_classes_allowed
            )
            self.dataset = JanusDataset(
                dataset,
                self.discrete_columns,
                self.device,
                config.dataset.preprocess_dataset,
            )

            if (
                config.dataset.save_preprocessed_dataset_path is not None
                or config.dataset.preprocess_dataset
            ):
                print("\t+ Saving preprocessed Janus data")
                torch.save(
                    (self.dataset, self.discrete_columns),
                    os.path.join(
                        config.dataset.save_preprocessed_dataset_path,
                        "janus_dataset_cat_{}_ns.pkl".format(
                            config.dataset.n_classes_allowed
                        ),
                    ),
                )

            if not config.dataset.preprocess_dataset:
                self.discrete_columns = None

        idx = torch.arange(0, self.dataset.n_samples, 1)
        self.fit_idx, self.val_idx = train_test_split(
            idx, test_size=0.10, random_state=42
        )

    def get_encoder_decoder(self):
        return (
            get_class(self.config.encoder.module)(
                data_dim=self.dataset.n_features,
                compress_dims=self.compress_dims,
                embedding_dim=self.embedding_dim,
            ),
            get_class(self.config.decoder.module)(
                data_dim=self.dataset.n_features,
                decompress_dims=self.decompress_dims,
                embedding_dim=self.embedding_dim,
            )
        )

    def run(self):

        model_config_path = "benchmark/configs/model"

        self.benchmark_results = dict()

        for model_target in self.config.models:
            model_name = model_target.split(".")[-1]

            temp_var = os.path.join(model_config_path, f"{model_name.lower()}.yaml")
            if os.path.exists(temp_var):
                print("+" * 60)
                model_config = omegaconf.OmegaConf.load(
                    temp_var
                )
            else:
                model_config = get_class("pythae.models.{}Config".format(model_target.split(".")[-1][:-1]))(
                    input_dim=(self.dataset.n_features,),
                    latent_dim=self.embedding_dim,
                    uses_default_encoder=False,
                    uses_default_decoder=False
                )

            print(f"\t+ Loading model: {model_name}")
            encoder, decoder = self.get_encoder_decoder()
            trainer = Trainer(
                logdir=self.config.logdir,
                device=self.device,
                dataset=self.dataset,
                fit_idx=self.fit_idx,
                val_idx=self.val_idx,
                batch_size=self.config.batch_size,
            )

            trainer.construct_model(
                model_target=model_target,
                encoder=encoder,
                decoder=decoder,
                model_config=model_config,
                categorical_columns=self.discrete_columns,
            )

            trainer.set_optimizer()
            trainer.set_scheduler()
            trainer.fit(epochs=self.config.epochs)

            del encoder
            del decoder
            del trainer
            gc.collect()
            if self.device.type == "cuda":
                print("\t+ Clearning GPU memory")
                torch.cuda.empty_cache()
            print("+" * 60)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.benchmark_results, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.benchmark_results = pickle.load(f)


@hydra.main(config_path="configs", config_name="benchmark")
def run(config):
    # print(config)
    benchmark = Benchmark(config=config)
    benchmark.run()


if __name__ == "__main__":
    run()
