import torch
from torch import nn

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams["backend"] = "Qt5Agg"

from mnist_utils import (
    AuxNetwork,
    GuidedVAE,
    get_mnist_loaders,
    mutual_information_is,
    plot_mutual_information,
)

args = {
    "n_epochs": 3,
    "lr": 0.002,
    "batch_size": 64,
    "inp_size": 28 * 28,
    "emb_sizes": [128, 64, 10],
    "lat_size": 2,
    "add_dropouts": True,
    "aux_network_path": "tests/mnist_guided_opt/weights/mnist_auxnet_weights.pkl"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# fit_loader, val_loader = get_mnist_loaders(args["batch_size"])

aux_network = AuxNetwork(args["inp_size"], args["emb_sizes"]).to(device)
aux_network.load_state_dict(
    state_dict=torch.load(
        args["aux_network_path"],
        map_location=device,
)["state_dict"])

guided_vae = GuidedVAE(
    inp_size=args["inp_size"],
    emb_sizes=args["emb_sizes"],
    lat_size=args["lat_size"],
    add_dropouts=args["add_dropouts"],
    aux_network=aux_network,
).to(device)

x = torch.randn(args["batch_size"], args["inp_size"], device=device)
xhat, mu, logvar = guided_vae(x)
print(
    xhat.size(),
    mu.size(),
    logvar.size()
)

# grid_size = 20
# step_size = 0.5
# plot_mutual_information(
#     guided_vae, grid_size, step_size, device
# )
