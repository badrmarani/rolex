seed=42
max_epochs=100
beta=0.01
batch_size=500

embedding_dim=2
compress_dims="[128,128]"
decompress_dims="[128,128]"

cmd="python experiments/janus/train_vae_janus.py \
  --seed=$seed \
  --batch_size $batch_size \
  --embedding_dim=$embedding_dim \
  --compress_dims=$compress_dims \
  --decompress_dims=$decompress_dims \
  --dataset_path=data\fulldataset.csv \
  --max_epochs=$max_epochs \
  --beta=$beta \
  --transform_data \
"
#   --cuda
echo $cmd
$cmd
echo $cmd
