import sys
import os

import torch
from mnist_utils import AuxNetwork, GuidedVAE
from base import Encoder, Decoder

from torch import nn
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import MNIST
import yaml


abspath = os.path.normpath("experiments/guided_opt/mnist_guided_opt")
args = yaml.safe_load(open(os.path.join(
    abspath,
    os.path.normpath("configs/mnist_configs.yml"),
)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(torch.flatten),
])

mnist_dataset_root = os.path.join(abspath, os.path.normpath("mnist_dataset/"))
fit_dataset = MNIST(mnist_dataset_root, train=True, download=True, transform=transform)
val_dataset = MNIST(mnist_dataset_root, train=False, download=True, transform=transform)

fit_loader = data.DataLoader(dataset=fit_dataset, batch_size=args["batch_size"], shuffle=True)
val_loader = data.DataLoader(dataset=val_dataset, batch_size=args["batch_size"], shuffle=True)
aux_net = AuxNetwork(
    args["lat_size"],
    args["emb_sizes"]["aux_net_bloc"],
    args["add_dropouts"],
    args["p"],
).to(device)

loss_fn = nn.NLLLoss()
optimizer = torch.optim.Adam(aux_net.parameters(), lr=args["lr"])

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

guided_vae.load_state_dict(
    torch.load(os.path.join(
        abspath,
        os.path.normpath("weights/mnist_vae_20_weights.pkl")
    ))["state_dict"]
)

for param in guided_vae.parameters():
    param.requires_grad = False


@torch.no_grad()
def make_all_positive(tensor):
    return torch.where(
        tensor < 0,
        -tensor,
        tensor,
    )

# aux_net.load_state_dict(
#     torch.load(os.path.join(
#         abspath,
#         os.path.normpath("weights/mnist_aux_net_reg_weights.pkl")
#     ))["state_dict"]
# )

for epoch in range(1, 1+args["n_epochs"]):
    aux_net.train()
    train_loss = 0.0
    correct = 0
    total = 0
    for i, batch in enumerate(fit_loader, 1):
        x, y = batch[0].to(device), batch[1].to(device)
        x = x.view(x.shape[0], -1)
        optimizer.zero_grad()
        
        with torch.no_grad():
            xhat, mu, logvar, sigma = guided_vae(x)
            z = guided_vae.rsample(mu, logvar)
        
        y_pred = aux_net(z)
        y_pred = torch.log_softmax(y_pred, dim=-1)

        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        _, y_pred = torch.max(y_pred.int().data, 1)
        y = y.to(torch.float)
        total += y.size(0)
        correct += (y_pred == y).sum().item()
        if i % args["log_interval"] == args["log_interval"] - 1:
            print(
                "[{}/{}, {}/{}] loss: {:.3f}, acc: {:3.3f}".format(
                    epoch,
                    args["n_epochs"],
                    i,
                    len(fit_loader),
                    train_loss / args["log_interval"],
                    100 * correct / total,
                )
            )
            train_loss = 0.0

print("training finished.")

aux_net.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for batch in val_loader:
        x, y = batch[0].to(device), batch[1].to(device)
        x = x.view(x.shape[0], -1)
        xhat, mu, logvar, sigma = guided_vae(x)
        z = guided_vae.rsample(mu, logvar)
        all_y_pred = aux_net(z)
        all_y_pred = torch.log_softmax(all_y_pred, dim=-1)

        _, y_pred = torch.max(all_y_pred.int().data, 1)
        y = y.to(torch.float)
        total += y.size(0)
        correct += (y_pred == y).sum().item()
    print(
        "accuracy of the network on the {} test images: {:3f}".format(
            len(val_loader.dataset), 100 * correct / total
        )
    )

    class_correct = [0] * 10
    class_total = [0] * 10
    for batch in val_loader:
        x, y = batch[0].to(device), batch[1].to(device)
        x = x.view(x.shape[0], -1)
        xhat, mu, logvar, sigma = guided_vae(x)
        z = guided_vae.rsample(mu, logvar)
        all_y_pred = aux_net(z)
        all_y_pred = torch.log_softmax(all_y_pred, dim=-1)

        _, y_pred = torch.max(all_y_pred.int(), 1)
        c = (y_pred == y).squeeze()
        for i in range(10):
            label = y[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

    for i in range(10):
        print("accuracy of {}: {:3f}".format(i, (class_correct[i] / class_total[i])))

weights_dirname = os.path.join(abspath, os.path.normpath("weights"))
os.makedirs(weights_dirname, exist_ok=True)
torch.save(
    {
        "state_dict": aux_net.to("cpu").state_dict(),
    },
    os.path.join(weights_dirname, os.path.normpath("mnist_aux_net_clf_weights.pkl")),
)
