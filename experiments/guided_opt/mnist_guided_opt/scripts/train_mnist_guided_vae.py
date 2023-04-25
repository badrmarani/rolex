from torchvision.datasets import MNIST
from torchvision import transforms
import os
import sys
abspath = os.path.normpath("experiments/guided_opt/mnist_guided_opt")
sys.path.append(abspath)

import torch
from torch import nn
from torch.utils import data

from mnist_utils import GuidedVAE, AuxNetwork, loss_function

from base import Encoder, Decoder

import yaml

args = yaml.safe_load(open(os.path.join(
    abspath,
    os.path.normpath("configs/mnist_configs.yml"),
)))

device = torch.device(args["device"])
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Lambda(torch.flatten),
# ])

# mnist_dataset_root = os.path.join(abspath, os.path.normpath("mnist_dataset/"))
# fit_dataset = MNIST(mnist_dataset_root, train=True,
#                     download=True, transform=transform)
# val_dataset = MNIST(mnist_dataset_root, train=False,
#                     download=True, transform=transform)

# fit_loader = data.DataLoader(
#     dataset=fit_dataset, batch_size=args["batch_size"], shuffle=True)
# val_loader = data.DataLoader(
#     dataset=val_dataset, batch_size=args["batch_size"], shuffle=True)

aux_net = AuxNetwork(
    args["lat_size"],
    args["emb_sizes"]["aux_net_bloc"],
    args["add_dropouts"],
    args["p"],
).to(device)
aux_net.load_state_dict(
    torch.load(os.path.join(
        abspath,
        os.path.normpath("weights/mnist_aux_net_clf_weights.pkl")
    ))["state_dict"]
)

mnist_dataset_root = os.path.join(abspath, os.path.normpath("mnist_dataset/"))
encoder = Encoder(inp_size=args["inp_size"], emb_sizes=args["emb_sizes"]["vae"], lat_size=args["lat_size"], add_dropouts=args["add_dropouts"], p=args["p"])
decoder = Decoder(lat_size=args["lat_size"], emb_sizes=args["emb_sizes"]["vae"], out_size=args["inp_size"], add_dropouts=args["add_dropouts"], p=args["p"])
guided_vae = GuidedVAE(
    encoder=encoder,
    decoder=decoder,
    inp_size=args["inp_size"],
    device=device,
    batch_size=args["batch_size"],
    lat_size=args["lat_size"],
    loss_beta=args["loss_beta"],
    transform_data=args["transform_data"],
    guide_latent_space=args["guide_latent_space"],
    save_transformed_data=args["save_transformed_data"],
    save_transformed_path=mnist_dataset_root,
).to(device)

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
