# ROLEX

ROLEX (**Ro**bust **L**atent Space **Ex**ploration) is a lightweight package designed to perform uncertainty-constrained optimization and help you apply uncertainty quantification techniques to generative models.

This package provides a multi-level API, including:

- ready-to-train VAE models on tabular datasets
- layers available for use in your networks
- optimization routines in the learned latent space

## Installation

```bash
conda create -n uncertainty python=3.10
conda activate uncertainty
pip install -e .
```

## Running an experiment

```bash
cmd="python .\experiments\dummy\learn_vae.py \
    --seed=42 \
    --embedding_dim=2 \
    --compress_dims=[128,128] \
    --decompress_dims=[128,128] \
    --dropout=0.3 \
    --precision=64 \
    --root=./data/dummy_data.csv \
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
echo $cmd
$cmd
```

### Hyperparameters Explanation

Here's a breakdown of the hyperparameters used in the command:

- `--seed`: Seed for random number generation.
- `--embedding_dim`: Dimension of the learned embeddings.
- `--compress_dims`: List of dimensions for each compression layer.
- `--decompress_dims`: List of dimensions for each decompression layer.
- `--dropout`: Dropout probability for preventing overfitting.
- `--precision`: Precision of floating-point numbers (32 or 64).
- `--root`: Path to the dataset (CSV format).
- `--max_epochs`: Maximum number of training epochs.
- `--batch_size`: Number of samples in each training batch.
- `--accelerator`: Hardware accelerator for training (e.g., "cpu", "gpu").
- `--lr`: Learning rate for optimizer.
- `--weight_decay`: Weight decay regularization strength.
- `--correlation_threshold`: Threshold for correlation matrix filtering.
- `--oversample_quantile`: Quantile for oversampling rare events.
- `--qq_threshold`: Threshold for quantile-quantile loss.
- `--beta_on_kld`: Coefficient for the KLD loss term.
- `--log_graph`: Flag to log computation graph.

## Implemented methods

### Layers

To date, the following layers are implemented:

- Bayesian Neural Networks
- Bayesian Flipout Neural Networks
- DropConnect
- Lipschitz Constrained Neural Networks

### Pre-Processing methods

To date, the following post-processing methods are implemented:

- Mode Specific Normalization

### Uncertainty metrics

To date, the following uncertainty metrics are implemented:

- Negative marginal log-likelihood
- Entropy
- Effective Sampling Size
- Mutual Information

### Optimization utilities

To date, the following optimization routines are implemented:

- Uncertainty-constrained gradient ascent (descent)
- Bayesian Optimization with uncertainty censoring
- Single-objective genetic algorithm optimization with uncertainty censoring
