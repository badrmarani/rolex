import torch
from torch import nn
from torch.utils import data

import matplotlib
import matplotlib.pyplot as plt

@torch.no_grad()
def save_latent_space(model: nn.Module, data: data.DataLoader, n_samples: int, device: torch.device):
    plt.figure()
    if n_samples is None:
        n_samples = data.dataset.data.size(0)
    y = data.dataset.targets[:n_samples].cpu().to(torch.float)
    x = data.dataset.data[:n_samples, ...].flatten(1).to(torch.float) / 255.
    z = model.rsample(*model.encoder(x)).cpu()
    plt.scatter(z[:, 0], z[:, 1], c=y, cmap="tab10")
    plt.colorbar()
    return plt

def loss_function(
    x: torch.Tensor,
    xhat: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    sigma: torch.Tensor,
    beta: float = 1.0,
    transform_info=None,
):
    rec_loss = []
    if transform_info is not None:
        for column_info in transform_info:
            for span_info in column_info:
                if span_info.activation_fn != "softmax":
                    ed = st + span_info.dim
                    std = sigma[st]
                    eq = x[:, st] - torch.tanh(xhat[:, st])
                    rec_loss.append((eq ** 2 / 2 / (std ** 2)).sum())
                    rec_loss.append(torch.log(std) * x.size()[0])
                    st = ed
                else:
                    ed = st + span_info.dim
                    rec_loss.append(nn.functional.cross_entropy(
                        xhat[:, st:ed], torch.argmax(x[:, st:ed], dim=-1), reduction="sum"))
                    st = ed
    else:
        eq = x - torch.tanh(xhat)
        rec_loss = eq**2/2/(sigma**2) + sigma.log()*x.size(0)
        rec_loss = rec_loss.sum(dim=-1)

    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.pow(2), dim=-1)
    loss = kld_loss.mean(dim=0) + beta * rec_loss.mean(dim=0)
    return (
        loss,
        kld_loss.mean(dim=0),
        rec_loss.mean(dim=0),
    )