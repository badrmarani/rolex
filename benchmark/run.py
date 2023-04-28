import torch
from torch.utils import data

from uqvae.quality_dataset import JanusDataset, prepare_dataset
from uqvae.trainer import Trainer
from uqvae.models import VAE, Encoder, Decoder

from sklearn.model_selection import train_test_split
import numpy as np
import random

generator = torch.Generator()
generator.manual_seed(42)
np.random.seed(42)


filename = "../data/fulldataset.csv"
dataset = prepare_dataset(filename)
janus_dataset = JanusDataset(dataset)

idx = torch.arange(0, janus_dataset.n_samples, 1)
fit_idx, val_idx = train_test_split(idx, test_size=0.10, random_state=42)

encoder = Encoder
decoder = Decoder
configs = {
    "data_dim": janus_dataset.n_features,
    "compress_dims": (128, 128),
    "decompress_dims": (128, 128),
    "embedding_dim": 100,
    "beta": 1.0,
    "cat_columns": None,
}

model = VAE(
    encoder=encoder,
    decoder=decoder,
    configs=configs,
)

print(model)

trainer = Trainer(model=model, logdir=".")
trainer.set_optimizer()
trainer.set_scheduler()


trainer.fit(janus_dataset, fit_idx, val_idx, batch_size=32, epochs=10)