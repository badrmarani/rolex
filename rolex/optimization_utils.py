import numpy as np
import torch
from botorch import fit_gpytorch_model
from botorch.acquisition import qExpectedImprovement
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from torch import nn
from tqdm import trange

from .uncertainty_metrics import mutual_information
from .utils import enable_dropout


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
                        [new_z.cpu().numpy(), pred_y.view(-1, 1).cpu().numpy()], axis=1
                    )
                ]

    out = torch.concatenate((z, y), dim=-1)
    if save_history:
        return out, np.concatenate(logs, axis=0)
    return out


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
    decoder.train()
    enable_dropout(decoder)

    if maximize:
        sgn = 1.0
    else:
        sgn = -1.0

    z.requires_grad = True
    if save_history:
        logs = []
    for _ in trange(n_steps, desc=f"Gradient {'ascent' if maximize else 'descent'}"):
        p = regressor(z)

        with torch.no_grad():
            if save_history:
                if save_history:
                    best_p_index = torch.argmax(p, dim=0)
                else:
                    best_p_index = torch.argmin(p, dim=0)
                best_p = p[best_p_index]
                logs += [
                    np.concatenate(
                        [z[best_p_index, ...].cpu().numpy(), best_p.cpu().numpy()],
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


class BaseProblem(Problem):
    def __init__(
        self,
        decoder,
        regressor,
        uncertainty_threshold_value,
        n_simulations,
        n_sampled_outcomes,
        no_uncertainty,
        lower_bound,
        upper_bound,
        embedding_dim,
        maximize,
        dtype,
        device,
    ) -> None:
        if not no_uncertainty:
            n_ieq_constr = 1
        else:
            n_ieq_constr = 0

        super().__init__(
            n_var=embedding_dim,
            n_obj=1,
            n_ieq_constr=n_ieq_constr,
            xl=np.array([lower_bound] * embedding_dim),
            xu=np.array([upper_bound] * embedding_dim),
        )

        self.no_uncertainty = no_uncertainty
        self.uncertainty_threshold_value = uncertainty_threshold_value
        self.n_simulations = n_simulations
        self.n_sampled_outcomes = n_sampled_outcomes
        self.decoder = decoder
        self.regressor = regressor
        self.maximize = maximize
        self.dtype = dtype
        self.device = device

    @torch.no_grad()
    def _evaluate(self, z, out, *args, **kwargs):

        if isinstance(self.regressor, nn.Module):
            z = torch.from_numpy(z).to(self.dtype)
            z = z.to(self.device)
            y = self.regressor(z).cpu().numpy()
        else:
            y = self.regressor(z)

        out["F"] = np.column_stack([y]) if self.maximize else np.column_stack([-y])

        if not self.no_uncertainty:
            z = torch.from_numpy(z).to(self.dtype).to(self.device)
            constraints = []
            with torch.no_grad():
                mi = (
                    mutual_information(
                        self.decoder,
                        z,
                        n_simulations=self.n_simulations,
                        n_sampled_outcomes=self.n_sampled_outcomes,
                        verbose=False,
                    )
                    .cpu()
                    .numpy()
                )
                constraints += [mi - self.uncertainty_threshold_value]

            out["G"] = np.column_stack(constraints)


def genetic_algorithm(
    decoder: nn.Module,
    regressor: nn.Module,
    uncertainty_threshold_value: float = None,
    n_simulations: int = None,
    n_sampled_outcomes: int = None,
    no_uncertainty: bool = False,
    save_history: bool = True,
    maximize: bool = True,
    lower_bound: float = -20.0,
    upper_bound: float = 20.0,
    embedding_dim: int = None,
    n_steps: int = 50,
    pop_size: int = 300,
    seed: int = None,
    verbose: bool = True,
    dtype: torch.dtype = None,
    device: torch.device = None,
):
    termination = get_termination("n_gen", n_steps)
    algorithm = GA(pop_size=pop_size, eliminate_duplicates=True)
    problem = BaseProblem(
        decoder,
        regressor,
        uncertainty_threshold_value,
        n_simulations,
        n_sampled_outcomes,
        no_uncertainty,
        lower_bound,
        upper_bound,
        embedding_dim,
        maximize,
        dtype,
        device,
    )

    return minimize(
        problem,
        algorithm,
        termination,
        seed=seed,
        save_history=save_history,
        verbose=verbose,
    )
