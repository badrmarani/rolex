import hydra
import torch

from dataset import JanusDataset
from benchmark import Benchmark

@hydra.main(config_path="configs", config_name="default_benchmark", version_base=None)
def run(config):
    benchmark = Benchmark(config)
    benchmark.load_data()
    benchmark.run()


if __name__ == "__main__":
    run()
