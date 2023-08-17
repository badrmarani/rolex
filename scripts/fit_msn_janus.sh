cmd="python .\experiments\janus\fit_msn_janus.py \
    --seed=42 \
    --precision=64 \
    --root=./data/janus.csv \
    --accelerator=gpu \
    --correlation_threshold=0.8 \
    --oversample_quantile=0.5 \
    --qq_threshold=0.96"
echo $cmd
$cmd