import os

seed = 42
batch_size = 500
embedding_dim = 2
compress_dims = "[128,128]"
decompress_dims = "[128,128]"
dataset_path = "data/fulldataset.csv"
max_epochs = 500
beta = 0.01
test_size = 0.05
transform_data = ["", "--transform_data"]
cuda = ["", "--cuda"]

semi_supervised_learning = ["", "--semi_supervised_learning"]
weight_type = ["uniform", "weight_decay"]

metric_loss_threshold = 0.0
metric_loss_beta = 1.0


cmd = f"""python experiments/janus/train_vae_janus.py \
--seed={seed} \
--batch_size={batch_size} \
--embedding_dim={embedding_dim} \
--compress_dims={compress_dims} \
--decompress_dims={decompress_dims} \
--dataset_path={dataset_path} \
--max_epochs={max_epochs} \
--beta={beta} \
--test_size={test_size} \
{transform_data[-1]} \
{cuda[-1]} \
"""

print(cmd)
os.system(cmd)
