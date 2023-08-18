import numpy as np
import torch
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from torch import nn

from ..metrics import mutual_information


class BaseProblem(Problem):
    """
    Base problem class for optimization using genetic algorithms.

    Parameters:
        decoder (nn.Module): The decoder network.
        regressor (nn.Module): The regressor network.
        uncertainty_threshold_value (float): The threshold value for uncertainty.
        n_simulations (int): Number of simulations.
        n_sampled_outcomes (int): Number of sampled outcomes.
        no_uncertainty (bool): Whether uncertainty is considered or not.
        lower_bound (float): Lower bound of optimization.
        upper_bound (float): Upper bound of optimization.
        embedding_dim (int): Dimension of the embedding.
        maximize (bool): Whether to maximize the objective.
        dtype (torch.dtype): Data type for tensors.
        device (torch.device): Device to perform computations.

    Attributes:
        no_uncertainty (bool): Whether uncertainty is considered or not.
        uncertainty_threshold_value (float): The threshold value for uncertainty.
        n_simulations (int): Number of simulations.
        n_sampled_outcomes (int): Number of sampled outcomes.
        decoder (nn.Module): The decoder network.
        regressor (nn.Module): The regressor network.
        maximize (bool): Whether to maximize the objective.
        dtype (torch.dtype): Data type for tensors.
        device (torch.device): Device to perform computations.
    """

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

        out["F"] = (
            np.column_stack([y]) if self.maximize else np.column_stack([-y])
        )

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
    """
    Single-objective genetic algorithm optimization with uncertainty censoring.

    Parameters:
        decoder (nn.Module): The decoder network.
        regressor (nn.Module): The regressor network.
        uncertainty_threshold_value (float, optional): The threshold value for uncertainty.
        n_simulations (int, optional): Number of simulations.
        n_sampled_outcomes (int, optional): Number of sampled outcomes.
        no_uncertainty (bool, optional): Whether uncertainty is considered or not.
        save_history (bool, optional): Whether to save optimization history.
        maximize (bool, optional): Whether to maximize the objective.
        lower_bound (float, optional): Lower bound for optimization. Default is -20.0.
        upper_bound (float, optional): Upper bound for optimization. Default is 20.0.
        embedding_dim (int, optional): Dimension of the embedding.
        n_steps (int, optional): Number of optimization steps.
        pop_size (int, optional): Population size for the genetic algorithm.
        seed (int, optional): Random seed for reproducibility.
        verbose (bool, optional): Whether to print optimization progress.
        dtype (torch.dtype, optional): Data type for tensors.
        device (torch.device, optional): Device to perform computations.

    Returns:
        result: The optimization result.
    """
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
