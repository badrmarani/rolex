import torch
from torch import nn

import yaml

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["backend"] = "Qt5Agg"

from mnist_utils import (
    Encoder,
    Decoder,
    AuxNetwork,
    GuidedVAE,
    get_mnist_loaders,
    loss_function,
    plot_mutual_information,
)

with open("tests/mnist_guided_opt/mnist_configs.yml", "r") as stream:
    args = yaml.safe_load(stream)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fit_loader, val_loader = get_mnist_loaders(args["batch_size"])

aux_network = AuxNetwork(
    args["inp_size"],
    args["emb_sizes"],
    args["add_dropouts"]
).to(device)
# aux_network.load_state_dict(
#     state_dict=torch.load(
#         args["aux_network_path"],
#         map_location=device,
#     )["state_dict"]
# )

encoder = Encoder(
    args["inp_size"],
    args["emb_sizes"]["vae"],
    args["lat_size"],
    args["add_dropouts"],
)

# encoder.seq[:7].load_state_dict(aux_network.seq[0][:6].state_dict())

decoder = Decoder(
    args["lat_size"],
    args["emb_sizes"]["vae"],
    args["inp_size"],
    args["add_dropouts"],
)

dumb_net = nn.Sequential(
    nn.Linear(args["lat_size"], 20), nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(20, 10), nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(10, 1),
)

guided_vae = GuidedVAE(
    encoder=encoder,
    decoder=decoder,
    n_gradient_steps=args["n_gradient_steps"],
    n_simulations=args["n_simulations"],
    n_sampled_outcomes=args["n_sampled_outcomes"],
    gradient_scale=args["gradient_scale"],
    uncertainty_threshold_value=args["uncertainty_threshold_value"],
    normalize_gradient=args["normalize_gradient"],
    aux_network=dumb_net,
).to(device)


loss_fn = loss_function
opt = torch.optim.Adam(guided_vae.parameters(), lr=args["lr"])
guided_vae.train()
for epoch in range(1, args["n_epochs"] + 1):
    running_loss = 0.0
    for i, batch in enumerate(fit_loader, 1):
        x, y = batch[0].to(device), batch[1].to(device)
        opt.zero_grad()
        xhat, mu, logvar = guided_vae(x)
        loss, kld, rec = loss_fn(x, xhat, mu, logvar)
        loss.backward()
        opt.step()

        running_loss += loss.item()
        if i % args["log_interval"] == args["log_interval"] - 1:
            print(
                "[{}/{}, {}/{}] running_loss: {:.3f}".format(
                    epoch,
                    args["n_epochs"],
                    i,
                    len(fit_loader),
                    running_loss / args["log_interval"],
                )
            )
print("training finished.")


grid_size = 20
step_size = 0.5
plot_mutual_information(
    model=guided_vae,
    grid_size=grid_size,
    step_size=step_size,
    n_simulations=args["n_simulations"],
    n_sampled_outcomes=args["n_sampled_outcomes"],
    device=device,
)
