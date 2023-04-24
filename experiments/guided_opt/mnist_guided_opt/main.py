import torch
from torch import nn

import yaml
import os
import matplotlib
import matplotlib.pyplot as plt

from mnist_utils import *

dirname = "experiments/guided_opt/mnist_guided_opt"
mnist_dataset_root = os.path.join(dirname, "mnist_dataset/")

with open(os.path.join(dirname, "mnist_configs.yml"), "r") as stream:
    args = yaml.safe_load(stream)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
# fit_loader, val_loader = get_mnist_loaders(mnist_dataset_root, args["batch_size"])

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
).to(device)

if not args["dataset_already_saved"]:
    fit_loader, val_loader = guided_vae.get_mnist_loaders(mnist_dataset_root)
else:
    fit_loader = torch.load(os.path.join(mnist_dataset_root, "transformed_fit_dataset.pkl"))
    val_loader = torch.load(os.path.join(mnist_dataset_root, "transformed_val_dataset.pkl"))


loss_fn = loss_function
optimizer = torch.optim.Adam(guided_vae.parameters(), lr=args["lr"])
guided_vae.train()
for epoch in range(1, args["n_epochs"] + 1):
    running_loss = 0.0
    for i, batch in enumerate(fit_loader, 1):
        x, y = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        xhat, mu, logvar, sigma = guided_vae(x)
        loss, kld, rec = loss_fn(x, xhat, mu, logvar, sigma, 2.0)
        loss.backward()
        optimizer.step()
        guided_vae.decoder.sigma.data.clamp_(0.01, 1.0)

        running_loss += loss.item()
        if i % args["log_interval"] == args["log_interval"] - 1:
            print(
                "[{}/{}, {}/{}] training loss: {:.3f}".format(
                    epoch,
                    args["n_epochs"],
                    i,
                    len(fit_loader),
                    running_loss / args["log_interval"],
                )
            )
            running_loss = 0.0
    
    with torch.no_grad():
        if not epoch%args["save_model_every_x_epochs"]:
            if args["guide_latent_space"]:
                grid_size = 20
                step_size = 0.5
                plt = plot_mutual_information(
                    model=guided_vae,
                    grid_size=grid_size,
                    step_size=step_size,
                    n_simulations=args["n_simulations"],
                    n_sampled_outcomes=args["n_sampled_outcomes"],
                    device=device,
                )
                
                mi_savedir = os.path.join(args["save_imgs_path"], "mutual_information/") 
                os.makedirs(mi_savedir, exist_ok=True)
                plt.savefig(
                    os.path.join(mi_savedir, "plot_mi_epoch_{}.jpg".format(epoch)),
                    dpi=300
                )
            if args["plot_latent_space"]:
                ls_savedir = os.path.join(args["save_imgs_path"], "latent_space/") 
                os.makedirs(ls_savedir, exist_ok=True)
                if args["lat_size"] > 2:
                    pass
                else:
                    plt = plot_latent_space(
                        model=guided_vae,
                        data=val_loader,
                        n_samples=1000,
                        device=device,
                    )

                plt.savefig(
                    os.path.join(ls_savedir, "plot_ls_epoch_{}.jpg".format(epoch)),
                    dpi=300
                )

            weights_dirname = os.path.join(dirname, "weights")
            os.makedirs(weights_dirname, exist_ok=True)
            torch.save(
                {
                    "state_dict": guided_vae.to("cpu").state_dict(),
                    "arguments": args,
                },
                os.path.join(weights_dirname, "mnist_vae_{}_weights.pkl".format(
                    epoch,
                )),
            )

        guided_vae.to(device)

print("training finished.") 
