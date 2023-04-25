from utils import get_dataset, get_stats

dirname = "experiments/vae_benchmark/all_models_logs"
dataset = get_dataset("data/fulldataset.csv", test_size=0.10, random_state=42)
eval = dataset["eval"]["pandas"].sample(500)
get_stats(dirname, eval, 1)
