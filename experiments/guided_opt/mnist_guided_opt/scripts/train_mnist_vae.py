import sys
import os
abspath = os.path.normpath("experiments/guided_opt/mnist_guided_opt")
sys.path.append(abspath)

import torch
from torch import nn

import yaml
import matplotlib
import matplotlib.pyplot as plt

from mnist_utils import *
from base import Encoder, Decoder

args = yaml.safe_load(open(os.path.join(
    abspath,
    os.path.normpath("configs/mnist_configs.yml"),
)))

mnist_dataset_root = os.path.join(abspath, os.path.normpath("mnist_dataset/"))

device = torch.device(args["device"])

# aux_network = AuxNetwork(
#     args["inp_size"],
#     args["emb_sizes"],
#     args["add_dropouts"]
# ).to(device)
# aux_network.load_state_dict(
#     state_dict=torch.load(
#         args["aux_network_path"],
#         map_location=device,
#     )["state_dict"]
# )

encoder = Encoder(
    inp_size=args["inp_size"],
    emb_sizes=args["emb_sizes"]["vae"],
    lat_size=args["lat_size"],
    add_dropouts=args["add_dropouts"],
    p=args["p"]
)

decoder = Decoder(
    lat_size=args["lat_size"],
    emb_sizes=args["emb_sizes"]["vae"],
    out_size=args["inp_size"],
    add_dropouts=args["add_dropouts"],
    p=args["p"]
)

guided_vae = GuidedVAE(
    encoder=encoder,
    decoder=decoder,
    guide_latent_space=args["guide_latent_space"],
    transform_data=args["transform_data"],
    save_transformed_data=args["save_transformed_data"],
    save_transformed_path=mnist_dataset_root,
    batch_size=args["batch_size"],
    inp_size=args["inp_size"],
    device=device,
    lat_size=args["lat_size"],
    loss_beta=args["loss_beta"],
).to(device)

if not args["dataset_already_saved"]:
    fit_loader, val_loader = guided_vae.get_mnist_loaders(mnist_dataset_root)
else:
    fit_loader = torch.load(os.path.join(mnist_dataset_root, "transformed_fit_dataset.pkl"))
    val_loader = torch.load(os.path.join(mnist_dataset_root, "transformed_val_dataset.pkl"))

loss_fn = loss_function
optimizer = torch.optim.Adam(guided_vae.parameters(), lr=args["lr"])

for epoch in range(1, args["n_epochs"] + 1):
    guided_vae.train_one_epoch(
        epoch=epoch,
        loss_fn=loss_fn,
        optimizer=optimizer,
        n_epochs=args["n_epochs"],
        log_interval=args["log_interval"],
        save_model_every_x_epochs=args["save_model_every_x_epochs"],
        guide_latent_space=args["guide_latent_space"],
        plot_latent_space=args["plot_latent_space"],
        save_imgs_path=args["save_imgs_path"],
        odirname=abspath,
    )
print("training finished.") 
