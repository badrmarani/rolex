import hydra
import torch

from dataset import JanusDataset
from benchmark import Benchmark

@hydra.main(config_path="configs", config_name="default_benchmark", version_base=None)
def run(config):
    benchmark = Benchmark(config)
    benchmark.prepare_data()

    for i in range 
        benchmark.run() # single configuration per model




if __name__ == "__main__":
    run()
