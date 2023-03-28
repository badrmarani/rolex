import matplotlib.pyplot as plt
from .losses import ELBOLoss
from torch.utils import data

def train_test_split(train_data, train_size: float = 0.8):
    train_set_size = int(len(train_data) * train_size)
    valid_set_size = len(train_data) - train_set_size
    train, val = data.random_split(train_data, [train_set_size, valid_set_size])
    return train, val

def train_one_epoch(
    epoch,
    model,
    loader,
    loss_fn,
    optimizer,
    device,
    log_intervall = 50
):
    model.train()
    train_loss = 0.0
    train_rec_loss, train_kld_loss = 0.0, 0.0
    for i, batch in enumerate(loader, start=1):
        x, _ = batch
        x = x.to(device)
        optimizer.zero_grad()
        xhat, mu, logvar = model(x)

        rec_loss, kld_loss = loss_fn(x, xhat, mu, logvar, mode="gaussian")
        loss = rec_loss + kld_loss
        train_loss += loss.item()
        train_rec_loss += rec_loss.item()
        train_kld_loss += kld_loss.item()
        loss.backward()
        optimizer.step()

        if not i%log_intervall:
            print(
                "train epoch: {} [{}/{} ({:.0f}%)] loss: {:.6f} kld_loss {:.6f} rec_loss {:.6f}"
                .format(
                    epoch, i*len(batch), len(batch)*len(loader),
                    100.*i/len(loader),
                    loss.item()/len(batch),
                    kld_loss.item()/len(batch),
                    rec_loss.item()/len(batch),
                )
            )

    print(
        "epoch: {} avg_loss: {:.6f} avg_kld_loss {:.6f} avg_rec_loss {:.6f}"
        .format(
            epoch,
            train_loss/len(loader.dataset),
            train_kld_loss/len(loader.dataset),
            train_rec_loss/len(loader.dataset),
        )
    )

    return model