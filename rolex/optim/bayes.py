import numpy as np
import torch
from botorch import fit_gpytorch_model
from botorch.acquisition import qExpectedImprovement
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import nn
from tqdm import trange

from ..metrics import mutual_information


def bayesian_optimization(
    decoder,
    regressor,
    z,
    y,
    n_steps,
    uncertainty_threshold_value: float = None,
    n_simulations: int = 100,
    n_sampled_outcomes: int = 10,
    no_uncertainty: bool = False,
    save_history: bool = True,
    lower_bound: float = -20.0,
    upper_bound: float = 20.0,
):
    """
    Bayesian Optimization with uncertainty censoring.

    Args:
        decoder: Decoder module.
        regressor: Regressor module or function.
        z (torch.Tensor): Initial latent representation.
        y (torch.Tensor): Initial predictions.
        n_steps (int): Number of optimization steps.
        uncertainty_threshold_value (float, optional): Threshold value for uncertainty-based filtering. Default is None.
        n_simulations (int, optional): Number of simulations for mutual information calculation. Default is 100.
        n_sampled_outcomes (int, optional): Number of sampled outcomes for mutual information calculation. Default is 10.
        no_uncertainty (bool, optional): Whether to use uncertainty-based filtering. Default is False.
        save_history (bool, optional): Whether to save optimization history. Default is True.
        lower_bound (float, optional): Lower bound for optimization. Default is -20.0.
        upper_bound (float, optional): Upper bound for optimization. Default is 20.0.

    Returns:
        torch.Tensor: Optimized latent representations and predictions.
        np.ndarray: Optimization history if save_history is True, else None.

    """
    n_training_points, embedding_dim = z.size()
    device = z.device

    if save_history:
        logs = []

    for _ in trange(n_steps, desc="Bayesian optimization"):
        train_x = z.detach().to(device)
        train_y = standardize(y).detach().to(device=device, dtype=z.dtype)
        train_y = train_y.view(-1, 1)

        gp = SingleTaskGP(train_x, train_y).to(device)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_model(mll)
        acq_fn = qExpectedImprovement(gp, best_f=0.1)
        bounds = torch.stack(
            [
                torch.ones(embedding_dim, dtype=z.dtype) * lower_bound,
                torch.ones(embedding_dim, dtype=z.dtype) * upper_bound,
            ]
        ).to(device)

        new_z, pred_y = optimize_acqf(
            acq_function=acq_fn,
            bounds=bounds,
            q=5,
            num_restarts=min(20, n_training_points),
            raw_samples=n_training_points,
            sequential=True,
            return_best_only=True,
        )

        new_z = new_z.to(z.dtype)
        pred_y = pred_y.view(-1, 1).to(dtype=z.dtype)

        if not no_uncertainty:
            mi = mutual_information(
                decoder,
                new_z,
                n_simulations=n_simulations,
                n_sampled_outcomes=n_sampled_outcomes,
                verbose=False,
            )
            index_below_threshold = mi < uncertainty_threshold_value
            n_below_threshold = index_below_threshold.int().sum()
            if n_below_threshold > 0:
                new_z = new_z[index_below_threshold][0]
                pred_y = pred_y[index_below_threshold][0]
            else:
                argmin_mi = mi.argmin()
                new_z = new_z[argmin_mi]
                pred_y = pred_y[argmin_mi]
        else:
            if len(new_z) > 1:
                new_z = new_z[0]
                pred_y = pred_y[0]

        new_z = new_z.view(1, -1).to(dtype=z.dtype)

        if isinstance(regressor, nn.Module):
            pred_y = regressor(new_z)
        else:
            pred_y = regressor(new_z.detach().cpu().numpy())
            pred_y = torch.from_numpy(pred_y).to(
                device=z.device,
                dtype=z.dtype,
            )

        pred_y = pred_y.view(1, -1).to(dtype=z.dtype)

        z = torch.cat(
            (z, new_z.view(-1, embedding_dim)),
            dim=0,
        ).to(torch.float32)
        y = torch.cat((y.view(-1, 1), pred_y), dim=0).to(z.dtype)

        if save_history:
            with torch.no_grad():
                logs += [
                    np.concatenate(
                        [new_z.cpu().numpy(), pred_y.view(-1, 1).cpu().numpy()],
                        axis=1,
                    )
                ]

    out = torch.concatenate((z, y), dim=-1)
    if save_history:
        return out, np.concatenate(logs, axis=0)
    return out
