for dropout in $(seq 0.6 0.1 1.0);
do
    cmd="python .\experiments\janus\learn_vae_janus_bench_dropout.py \
        --seed=42 \
        --embedding_dim=2 \
        --compress_dims=[128,128] \
        --decompress_dims=[128,128] \
        --dropout=$dropout \
        --precision=64 \
        --root=./data/janus.csv \
        --max_epochs=100 \
        --batch_size=2048 \
        --accelerator=gpu \
        --lr=0.001 \
        --weight_decay=0.00001 \
        --correlation_threshold=0.8 \
        --oversample_quantile=0.5 \
        --qq_threshold=0.96 \
        --beta_on_kld=0.1 \
        --log_graph"
    echo $dropout
    echo $cmd
    $cmd
done
