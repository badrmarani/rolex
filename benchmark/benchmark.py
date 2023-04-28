import torch
import pickle

from .config import BenchmarkConfig

class VAEBenchmark:
    def __init__(self, config=BenchmarkConfig()):        
        self.config = config

    def run(self):
        self.benchmark_results = dict()

        for model_name in self.config.inputs_space:
            pass

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.benchmark_results, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.benchmark_results = pickle.load(f)
