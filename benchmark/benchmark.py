import torch
import pickle

import importlib

from types import SimpleNamespace

from uqvae.quality_dataset import JanusDataset, preprocess_janus_dataset
from uqvae.trainer import Trainer
from uqvae.models import VAE, Encoder, Decoder

import warnings
warnings.filterwarnings("ignore")

class Benchmark:
    def __init__(self, config):        
        self.config = config
        device = torch.device(self.config.device)

        print("Preprocessing Janus data.")
        dataset, self.discrete_columns = preprocess_janus_dataset(config.filename, config.n_classes_allowed)
        self.dataset = JanusDataset(
            dataset,
            self.discrete_columns,
            device,
            config.preprocess_dataset
        )


    def run(self):
        print(self.dataset.x.size())
        print(self.dataset.y.size())

        self.benchmark_results = dict()
        for model_name in self.config.models:
            pass
            

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.benchmark_results, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.benchmark_results = pickle.load(f)


config = SimpleNamespace(**{
    "filename": "data/fulldataset.csv",
    "n_classes_allowed": 2,
    "device": "cpu",
    "preprocess_dataset": True,
    "models": ["VAE",],
})


benchmark = Benchmark(config=config)
benchmark.run()