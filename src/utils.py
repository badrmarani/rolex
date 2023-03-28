import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils import data

from .losses import ELBOLoss
import pandas as pd


def plot_latent_space(
    epoch: int,
    model: nn.Module,
    device: torch.device,
    df: pd.DataFrame,
    n_samples: int,
    save: bool,
):
    fig, ax = plt.subplots(1)

    nonan_df0 = df[~df["target_000"].isna()]
    x = torch.from_numpy(
        nonan_df0[[col for col in nonan_df0.columns if col.startswith("data")]].values[
            :n_samples, :
        ]
    ).to(device=device, dtype=torch.float)

    y = nonan_df0["target_000"].values[:n_samples]

    xhat, _, _ = model(x)
    xhat = xhat.detach().cpu()

    if xhat.shape[-1] != 2:
        from sklearn.manifold import TSNE

        n_components = 2
        tsne = TSNE(n_components)

        xhat = xhat.numpy()
        xhat = tsne.fit_transform(xhat)

    plt.scatter(
        xhat[:, 0],
        xhat[:, 1],
        c=y,
        s=10,
    )
    plt.xlabel("$Z_1$")
    plt.ylabel("$Z_2$")
    plt.title("Epoch {}".format(epoch))
    plt.colorbar()
    if save is not None:
        plt.savefig(save, format="png")


def train_test_split(train_data, train_size: float = 0.8):
    train_set_size = int(len(train_data) * train_size)
    valid_set_size = len(train_data) - train_set_size
    train, val = data.random_split(train_data, [train_set_size, valid_set_size])
    return train, val


def train_one_epoch(epoch, model, loader, loss_fn, optimizer, device, log_intervall=50):
    model.train()
    train_loss = 0.0
    train_rec_loss, train_kld_loss = 0.0, 0.0
    for i, batch in enumerate(loader, start=1):
        x, _ = batch
        x = x.to(device)
        optimizer.zero_grad()
        xhat, mu, logvar = model(x)

        loss, rec_loss, kld_loss = loss_fn(x, xhat, mu, logvar, mode="gaussian")
        train_loss += loss.item()
        train_rec_loss += rec_loss.item()
        train_kld_loss += kld_loss.item()
        loss.backward()
        optimizer.step()

        if not i % log_intervall:
            print(
                "train epoch: {} [{}/{} ({:.0f}%)] loss: {:.6f} kld_loss {:.6f} rec_loss {:.6f}".format(
                    epoch,
                    i * len(batch),
                    len(batch) * len(loader),
                    100.0 * i / len(loader),
                    loss.item() / len(batch),
                    kld_loss.item() / len(batch),
                    rec_loss.item() / len(batch),
                )
            )

    print(
        "epoch: {} avg_loss: {:.6f} avg_kld_loss {:.6f} avg_rec_loss {:.6f}".format(
            epoch,
            train_loss / len(loader.dataset),
            train_kld_loss / len(loader.dataset),
            train_rec_loss / len(loader.dataset),
        )
    )

    return model


class SaveBestModel:
    pass
