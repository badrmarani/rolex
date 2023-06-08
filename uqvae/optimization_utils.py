import torch
from torch import nn
from tqdm import trange

from .metrics import mutual_information
from .utils import enable_dropout


def gradient_optimization(
    decoder: nn.Module,
    property_network: nn.Module,
    z: torch.Tensor,
    n_gradient_steps: int,
    gradient_scale: float,
    normalize_gradients: bool,
    uncertainty_threshold_value: float = None,
    n_simulations: int = 100,
    n_sampled_outcomes: int = 10,
    no_uncertainty: bool = False,
    track_logs: bool = True,
    maximize: bool = True,
    lower_bound: float = -20.0,
    upper_bound: float = 20.0,
):
    decoder.train()
    enable_dropout(decoder)

    if maximize:
        sgn = 1.0
    else:
        sgn = -1.0

    z.requires_grad = True
    if track_logs:
        logs_z = [z]
    for _ in trange(n_gradient_steps, desc="gradient ascent"):
        p = property_network(z)
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

        # TODO: generalize it to embedding_dim > 2
        z.data.clip_(lower_bound, upper_bound)
        if track_logs:
            logs_z.append(z)
    if track_logs:
        return z, logs_z
    return z
