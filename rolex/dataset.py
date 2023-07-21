import argparse
import functools

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from ctgan.data_transformer import DataTransformer
from sklearn.model_selection import train_test_split
from torch import utils
from torch.utils import data
from torchvision.datasets import MNIST

from .utils import seed_everything


class DataModule:
    def __init__(self, config, data_weighter=None, valid=False) -> None:
        super().__init__()
        self.config = config
        self.data_weighter = data_weighter
        self.valid = valid

        self.generator = torch.Generator()
        self.generator.manual_seed(config.seed)

    def prepare(self, train: bool = True):
        raise NotImplemented

    def setup(self, root):
        self.train_data, self.train_targets = self.prepare(train=True, root=root)
        self.train_dataset = utils.data.TensorDataset(
            self.train_data, self.train_targets
        )

        if self.valid:
            self.valid_data, self.valid_targets = self.prepare(train=False, root=root)
            self.valid_dataset = utils.data.TensorDataset(
                self.valid_data, self.valid_targets
            )

        self.set_weights()

    def set_weights(self):
        train_weights = self.data_weighter.weighting_function(self.train_targets)
        self.train_sampler = utils.data.WeightedRandomSampler(
            weights=train_weights, num_samples=len(self.train_targets), replacement=True
        )

        if self.valid:
            valid_weights = self.data_weighter.weighting_function(self.valid_targets)
            self.valid_sampler = utils.data.WeightedRandomSampler(
                weights=valid_weights,
                num_samples=len(self.valid_targets),
                replacement=True,
            )

    def train_dataloader(self):
        return utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.train_batch_size,
            num_workers=self.config.num_workers,
            sampler=self.train_sampler,
            # shuffle=self.config.valid_shuffle,
            drop_last=self.config.train_drop_last,
            pin_memory=self.config.pin_memory,
            worker_init_fn=seed_everything,
            generator=self.generator,
        )

    def valid_dataloader(self):
        if self.valid:
            return utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.config.valid_batch_size,
                num_workers=self.config.num_workers,
                sampler=self.valid_sampler,
                # shuffle=self.config.valid_shuffle,
                drop_last=self.config.valid_drop_last,
                pin_memory=self.config.pin_memory,
                worker_init_fn=seed_everything,
                generator=self.generator,
            )
        return None


class DataWeighter:
    def __init__(self, config):

        print("config.weight_type ->", config.weight_type)

        if config.weight_type in ["uniform"]:
            self.weighting_function = functools.partial(DataWeighter.uniform_weights)
        elif config.weight_type in ["property_values"]:
            self.weighting_function = functools.partial(
                DataWeighter.property_values_weights, rank_weight_k=config.rank_weight_k
            )
        elif config.weight_type in ["rank"]:
            self.weighting_function = functools.partial(
                DataWeighter.rank_weights, rank_weight_k=config.rank_weight_k
            )

        self.weight_type = config.weight_type

    @staticmethod
    def normalize_weights(weights: np.array):
        return weights / np.mean(weights)

    @staticmethod
    def uniform_weights(properties: torch.Tensor):
        return torch.ones_like(properties)

    @staticmethod
    def property_values_weights(properties: torch.Tensor):
        return properties

    @staticmethod
    def rank_weights(properties: torch.Tensor, rank_weight_k: float):
        ranks = torch.argsort(torch.argsort(-1 * properties))
        weights = 1.0 / (rank_weight_k * len(properties) + ranks)
        return weights


# class JANUSDataModule(pl.LightningDataModule):
#     def __init__(self, hparams, data_weighter) -> None:
#         super().__init__()

#         self.save_hyperparameters(hparams)
#         self.dataset_path = hparams.dataset_path
#         self.test_size = hparams.test_size
#         self.batch_size = hparams.batch_size
#         self.transform_data = hparams.transform_data
#         self.data_weighter = data_weighter

#         self.specific_setup()
#         self.production_dataset_preprocess()
#         if self.hparams.semi_supervised_learning:
#             self.set_weights()
#         self.make_tensor_dataset()

#     @staticmethod
#     def add_model_specific_args(parent_parser):
#         data_group = parent_parser.add_argument_group(title="data")
#         data_group.add_argument(
#             "--dataset_path", type=str, required=True, help="path to csv file"
#         )
#         data_group.add_argument("--batch_size", type=int, default=500)
#         data_group.add_argument(
#             "--test_size",
#             type=float,
#             default=0.05,
#             help="Fraction of val data.",
#         )
#         data_group.add_argument("--transform_data", default=False, action="store_true")
#         data_group.add_argument(
#             "--semi_supervised_learning", default=False, action="store_true"
#         )
#         return parent_parser

#     def production_dataset_preprocess(self):
#         if self.transform_data:
#             print("Transform data")
#             self.data_transformer = DataTransformer()
#             self.data_transformer.fit(self.data_train, ())
#             self.data_train = torch.from_numpy(
#                 self.data_transformer.transform(self.data_train).astype("float32")
#             )
#             self.data_val = torch.from_numpy(
#                 self.data_transformer.transform(self.data_val).astype("float32")
#             )
#         else:
#             print("No data tranformation")
#             self.data_train = torch.from_numpy(self.data_train.values.astype("float32"))
#             self.data_val = torch.from_numpy(self.data_val.values.astype("float32"))
#         self.data_dim = self.data_train.size(-1)

#     def setup(self, stage: str):
#         pass

#     def set_weights(self, data, properties):
#         weights = DataWeighter.uniform_weights(properties.values.astype("float32"))
#         self.sampler = torch.utils.data.WeightedRandomSampler(
#             weights=weights,
#             num_samples=len(weights),
#             replacement=True,
#         )

#     def specific_setup(self):
#         if not self.hparams.semi_supervised_learning:
#             data, _, _, _ = load(self.dataset_path)
#             indx_train, indx_val = tensor_train_test_split_indx(
#                 data,
#                 test_size=self.test_size,
#                 random_state=42,
#                 shuffle=False,
#             )
#             self.data_train = data.iloc[indx_train]
#             self.data_val = data.iloc[indx_val]
#         else:
#             data, _, _, target = load(self.dataset_path)
#             mask = np.where(target.notna().all(1))[0]
#             data = data.iloc[mask, :]
#             # right now, we're focusing on the first quality target.
#             target = target.iloc[mask, 0]
#             indx_train, indx_val = tensor_train_test_split_indx(
#                 data,
#                 test_size=self.test_size,
#                 random_state=42,
#                 shuffle=False,
#             )
#             self.data_train = data.iloc[indx_train]
#             self.target_train = torch.from_numpy(
#                 target.iloc[indx_train].values.astype("float32")
#             )
#             self.data_val = data.iloc[indx_val]
#             self.target_val = torch.from_numpy(
#                 target.iloc[indx_val].values.astype("float32")
#             )

#     def make_tensor_dataset(self):
#         self.train_dataset = data.TensorDataset(self.data_train)
#         self.valid_dataset = data.TensorDataset(self.data_val)
#         if self.hparams.semi_supervised_learning:
#             self.train_dataset = data.TensorDataset(self.data_train, self.target_train)
#             self.valid_dataset = data.TensorDataset(self.data_val, self.target_val)

#     def train_dataloader(self):
#         return data.DataLoader(
#             self.train_dataset,
#             batch_size=self.batch_size,
#             num_workers=NUM_WORKERS,
#             shuffle=False,
#             worker_init_fn=seed_everything,
#             generator=GENERATOR,
#             sampler=(
#                 self.train_sampler if self.hparams.semi_supervised_learning else None
#             ),
#         )

#     def valid_dataloader(self):
#         return data.DataLoader(
#             self.valid_dataset,
#             batch_size=self.batch_size,
#             num_workers=NUM_WORKERS,
#             shuffle=False,
#             worker_init_fn=seed_everything,
#             generator=GENERATOR,
#             sampler=(
#                 self.valid_sampler if self.hparams.semi_supervised_learning else None
#             ),
#         )

#     def set_weights(self):
#         weights_train = self.data_weighter.weighting_function(self.target_train)
#         weights_val = self.data_weighter.weighting_function(self.target_val)

#         self.train_sampler = torch.utils.data.WeightedRandomSampler(
#             weights=weights_train,
#             num_samples=len(weights_train),
#             replacement=True,
#         )
#         self.valid_sampler = torch.utils.data.WeightedRandomSampler(
#             weights=weights_val,
#             num_samples=len(weights_val),
#             replacement=True,
#         )
