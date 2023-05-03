import hydra
from benchmark import Benchmark

@hydra.main(config_path="configs", config_name="benchmark")
def run(config):
    # print(config)
    benchmark = Benchmark(config=config)
    benchmark.run()


if __name__ == "__main__":
    run()
