import numpy as np
import torch
from torch import nn
from tqdm import trange

from ..metrics import mutual_information
from ..metrics.utils import enable_dropout


def gradient_optimization(
    decoder: nn.Module,
    regressor: nn.Module,
    z: torch.Tensor,
    n_steps: int,
    gradient_scale: float,
    normalize_gradients: bool,
    uncertainty_threshold_value: float = None,
    n_simulations: int = 100,
    n_sampled_outcomes: int = 10,
    no_uncertainty: bool = False,
    save_history: bool = True,
    maximize: bool = True,
    lower_bound: float = -20.0,
    upper_bound: float = 20.0,
):
    """
    Uncertainty-constrained gradient ascent (descent).

    Args:
        decoder (nn.Module): The decoder model.
        regressor (nn.Module): The regressor model.
        z (torch.Tensor): The input latent tensor to be optimized.
        n_steps (int): Number of optimization steps.
        gradient_scale (float): Scale factor for the gradient update.
        normalize_gradients (bool): If True, normalize gradients.
        uncertainty_threshold_value (float, optional): Threshold for uncertainty-based update.
        n_simulations (int): Number of simulations for uncertainty computation.
        n_sampled_outcomes (int): Number of sampled outcomes for uncertainty computation.
        no_uncertainty (bool): If True, ignore uncertainty-based update.
        save_history (bool): If True, save optimization history.
        maximize (bool): If True, perform gradient ascent, else gradient descent.
        lower_bound (float, optional): Lower bound for optimization. Default is -20.0.
        upper_bound (float, optional): Upper bound for optimization. Default is 20.0.


    Returns:
        torch.Tensor: The optimized latent tensor 'z'.

    """
    decoder.eval()
    enable_dropout(decoder)

    if maximize:
        sgn = 1.0
    else:
        sgn = -1.0

    z.requires_grad = True
    if save_history:
        logs = []
    for _ in trange(
        n_steps, desc=f"Gradient {'ascent' if maximize else 'descent'}"
    ):
        p = regressor(z)

        with torch.no_grad():
            if save_history:
                if maximize:
                    best_p_index = torch.argmax(p, dim=0)
                else:
                    best_p_index = torch.argmin(p, dim=0)
                best_p = p[best_p_index]
                logs += [
                    np.concatenate(
                        [
                            z[best_p_index, ...].cpu().numpy(),
                            best_p.cpu().numpy(),
                        ],
                        axis=1,
                    )
                ]

        gradient = torch.autograd.grad(
            outputs=p,
            inputs=z,
            grad_outputs=torch.ones_like(p, device=z.device),
            retain_graph=False,
        )[0]
        if normalize_gradients:
            gradient.div_(gradient.norm(2))
        updated_z = z + sgn * gradient * gradient_scale
        if no_uncertainty:
            z = updated_z
        else:
            mi = mutual_information(
                decoder=decoder,
                latent_sample=updated_z,
                n_simulations=n_simulations,
                n_sampled_outcomes=n_sampled_outcomes,
                verbose=False,
            )
            z = torch.where(
                (mi <= uncertainty_threshold_value).unsqueeze(-1),
                updated_z,
                z,
            )

        z.data.clip_(lower_bound, upper_bound)
    if save_history:
        logs = np.concatenate(logs, axis=0)
        return z, logs
    return z
