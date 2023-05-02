import torch
from torch.utils import data

from uqvae.quality_dataset import JanusDataset, preprocess_janus_dataset
from uqvae.trainer import Trainer
from uqvae.models import VAE, Encoder, Decoder

from sklearn.model_selection import train_test_split
import numpy as np
import random

import os
import yaml
import importlib
generator = torch.Generator()
generator.manual_seed(42)
np.random.seed(42)


filename = "data/fulldataset.csv"
dataset, discrete_columns = preprocess_janus_dataset(filename, n_classes_allowed=2)

print(discrete_columns)
print(dataset[discrete_columns])
import sys; sys.exit()

dataset = JanusDataset(dataset, discrete_columns, preprocess_dataset=False)


configs = {
    "data_dim": dataset.n_features,
    "compress_dims": (128, 128//2, 10),
    "decompress_dims": (10, 128//2, 128),
    "embedding_dim": 5,
}


idx = torch.arange(0, dataset.n_samples, 1)
fit_idx, val_idx = train_test_split(idx, test_size=0.10, random_state=42)


module = importlib.import_module("uqvae.models")

for config_model in os.listdir("benchmark/configs"):

    model_name = config_model.split(".")[0].upper()
    print(model_name)

    config_model = dict(
        yaml.safe_load(
            open(f"benchmark/configs/{config_model}")
        ),
        **configs,
    )

    encoder = Encoder
    decoder = Decoder
    discrete_columns_idx = [i for i, col in enumerate(discrete_columns)]
    if model_name.startswith("T"):
        model = getattr(module, "VAE")(
            encoder=encoder, decoder=decoder,
            configs=config_model,
            categorical_columns=discrete_columns_idx,
        )
    else:
        model = getattr(module, model_name)(
            encoder=encoder, decoder=decoder,
            configs=config_model,
            categorical_columns=discrete_columns_idx,
        )



    # logdir = f"results/logs/{model_name}"
    # os.makedirs(logdir, exist_ok=True)

    # trainer = Trainer(model=model, logdir=logdir)
    # trainer.set_optimizer(lr=1e-5, betas=(0.9, 0.999), eps=1e-8)
    # trainer.set_scheduler(step=10, gamma=0.5)
    # trainer.fit(dataset, fit_idx, val_idx, batch_size=64, epochs=10)    

    # del encoder
    # del decoder
    # del model
    