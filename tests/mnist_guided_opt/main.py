import torch
from torch import nn

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["backend"] = "Qt5Agg"

from mnist_utils import (
    AuxNetwork,
    GuidedVAE,
    get_mnist_loaders,
    plot_mutual_information,
)

args = {
    "n_epochs": 3,
    "lr": 0.002,
    "batch_size": 64,
    "inp_size": 28 * 28,
    "emb_sizes": [128, 64, 10],
    "lat_size": 2,
    "beta": 1.0,
    "add_dropouts": True,
    "n_simulations": 10,
    "n_sampled_outcomes": 10,
    "aux_network_path": "tests/mnist_guided_opt/weights/mnist_auxnet_weights.pkl",
    "n_gradient_steps": 20,
    "gradient_scale": 0.1,
    "normalize_gradient": True,
    "uncertainty_threshold_value": 1000,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# fit_loader, val_loader = get_mnist_loaders(args["batch_size"])

aux_network = AuxNetwork(args["inp_size"], args["emb_sizes"]).to(device)
aux_network.load_state_dict(
    state_dict=torch.load(
        args["aux_network_path"],
        map_location=device,
    )["state_dict"]
)

guided_vae = GuidedVAE(
    inp_size=args["inp_size"],
    emb_sizes=args["emb_sizes"],
    lat_size=args["lat_size"],
    n_gradient_steps=args["n_gradient_steps"],
    n_simulations=args["n_simulations"],
    n_sampled_outcomes=args["n_sampled_outcomes"],
    gradient_scale=args["gradient_scale"],
    uncertainty_threshold_value=args["uncertainty_threshold_value"],
    add_dropouts=args["add_dropouts"],
    normalize_gradient=args["normalize_gradient"],
    aux_network=args["aux_network"],
).to(device)

# x = torch.randn(args["batch_size"], args["inp_size"], device=device)
# xhat, mu, logvar = guided_vae(x)
# print(
#     xhat.size(),
#     mu.size(),
#     logvar.size()
# )

# grid_size = 20
# step_size = 0.5
# plot_mutual_information(
#     model=guided_vae,
#     grid_size=grid_size,
#     step_size=step_size,
#     n_simulations=args["n_simulations"],
#     n_sampled_outcomes=args["n_sampled_outcomes"],
#     device=device,
# )
