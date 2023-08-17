import numpy as np
import torch
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from torch import nn

from ..metrics import mutual_information


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
