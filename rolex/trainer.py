import os
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import torch
from torch import distributions, nn, utils
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange


@dataclass
class Trainer:
    model: nn.Module
    config = None

    signature = datetime.now().strftime("_%Y%m%d_%H%M%S")

    def _init_trainer(self, train_loader, valid_loader, config, dtype, **kwargs):
        self.dtype = dtype
        self.config = config
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.device = torch.device(self.config.device)
        self.model = self.model.to(self.device)

        self.optimizer, self.scheduler = self.model.configure_optimizer(self.config)
        self.writer = SummaryWriter(
            log_dir=self.config.log_dir,
            filename_suffix=self.signature,
        )

    def _step(self):
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

    def train_step(self, loader, current_epoch):
        for data in loader:
            self.optimizer.zero_grad()
            loss, log = self.model(
                data,
                betas=(self.config.beta_kld, self.config.beta_reg),
            )
            loss.backward()
            self._step()
        return log

    @torch.no_grad()
    def validate(self, loader, current_epoch):
        for data in loader:
            _, log = self.model(
                data,
                betas=(self.config.beta_kld, self.config.beta_reg),
            )
        return log

    def write_results(self, writer, log, mode, current_epoch):
        for k, v in log.items():
            if v is not None:
                writer.add_scalar(f"{k}/{mode}", v, current_epoch)

    def fit(self, train_loader, valid_loader=None, config=None, **kwargs):
        if config is None:
            raise ValueError()
        self._init_trainer(train_loader, valid_loader, config, **kwargs)
        for i in trange(self.config.epochs):
            train_log = self.train_step(self.train_loader, i)
            if self.valid_loader is not None:
                valid_log = self.validate(self.valid_loader, i)

            self.write_results(self.writer, train_log, mode="train", current_epoch=i)
            self.write_results(self.writer, valid_log, mode="valid", current_epoch=i)

        self.writer.flush()
        self.writer.close()

        if config.save:
            self.save(self.config.log_dir)

    def save(self, model_path):
        sched = None
        if self.scheduler is not None:
            sched = self.scheduler.state_dict()

        state_dict = dict(
            model=self.model.state_dict(),
            optim=self.optimizer.state_dict(),
            sched=sched,
            config=self.config,
        )

        torch.save(state_dict, os.path.join(model_path, "model.ckpt"))

    def optimize(self, regressor):
        if self.config.optimization.optimize:
            z_init = torch.empty(
                30, self.config.embedding_dim, device=self.device
            ).uniform_(
                self.config.optimization.lower_bound,
                self.config.optimization.upper_bound,
            )

            name = self.config.optimization.optimization_method

            decoder = self.model.decoder

            if self.model.regressor is not None:
                regressor = self.model.regressor
            else:
                regressor = regressor.predict

            n_steps = self.config.optimization.n_steps
            gradient_scale = self.config.optimization.gradient_scale
            normalize_gradients = self.config.optimization.normalize_gradients
            uncertainty_threshold_value = (
                self.config.optimization.uncertainty_threshold_value
            )
            n_simulations = self.config.optimization.n_simulations
            n_sampled_outcomes = self.config.optimization.n_sampled_outcomes
            no_uncertainty = self.config.optimization.no_uncertainty
            save_history = self.config.optimization.save_history
            maximize = self.config.optimization.maximize
            lower_bound = self.config.optimization.lower_bound
            upper_bound = self.config.optimization.upper_bound

            if name == "gradient_optimization":
                if not isinstance(regressor, nn.Module):
                    raise ValueError(
                        "The regressor is not differentiable to be able to compute the gradients."
                    )

                from rolex.optimization_utils import gradient_optimization

                last_z, logs = gradient_optimization(
                    decoder=decoder,
                    regressor=regressor,
                    z=z_init,
                    n_steps=n_steps,
                    gradient_scale=gradient_scale,
                    normalize_gradients=normalize_gradients,
                    uncertainty_threshold_value=uncertainty_threshold_value,
                    n_simulations=n_simulations,
                    n_sampled_outcomes=n_sampled_outcomes,
                    no_uncertainty=no_uncertainty,
                    save_history=save_history,
                    maximize=maximize,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                )
            elif name == "bayesian_optimization":
                from rolex.optimization_utils import bayesian_optimization

                with torch.no_grad():
                    if isinstance(regressor, nn.Module):
                        y_init = regressor(z_init)
                    else:
                        y_init = regressor(z_init.cpu().numpy())
                        y_init = torch.from_numpy(y_init).to(
                            device=self.device,
                            dtype=self.dtype,
                        )

                last_z, logs = bayesian_optimization(
                    decoder=decoder,
                    regressor=regressor,
                    z=z_init,
                    y=y_init,
                    n_steps=n_steps,
                    uncertainty_threshold_value=uncertainty_threshold_value,
                    n_simulations=n_simulations,
                    n_sampled_outcomes=n_sampled_outcomes,
                    no_uncertainty=no_uncertainty,
                    save_history=save_history,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                )
            elif name == "genetic_algorithm":
                from rolex.optimization_utils import genetic_algorithm

                logs = genetic_algorithm(
                    decoder=decoder,
                    regressor=regressor,
                    uncertainty_threshold_value=uncertainty_threshold_value,
                    n_simulations=n_simulations,
                    n_sampled_outcomes=n_sampled_outcomes,
                    no_uncertainty=no_uncertainty,
                    save_history=save_history,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    maximize=maximize,
                    n_steps=n_steps,
                    embedding_dim=self.config.embedding_dim,
                    pop_size=self.config.optimization.pop_size,
                    verbose=self.config.optimization.verbose,
                    dtype=self.dtype,
                    seed=42,
                    device=self.device,
                )
            else:
                raise ValueError(f"{name} is not implemented")
            np.save(os.path.join(self.config.log_dir, "optim_logs.npy"), logs)
