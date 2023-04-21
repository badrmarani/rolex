import os

import torch
from mnist_utils import AuxNetwork
from torch import nn
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import MNIST

log_interval = 100

args = {
    "n_epochs": 3,
    "lr": 0.002,
    "batch_size": 64,
    "inp_size": 28 * 28,
    "emb_sizes": [128, 64, 10],
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(torch.flatten),
    ]
)
fit_loader = data.DataLoader(
    dataset=MNIST(
        "tests/mnist_guided_opt/mnist_dataset",
        train=True,
        download=True,
        transform=transform,
    ),
    batch_size=args["batch_size"],
    shuffle=True,
)


net = AuxNetwork(args["inp_size"], args["emb_sizes"]).to(device)

loss_fn = nn.NLLLoss()
opt = torch.optim.Adam(net.parameters(), lr=args["lr"])

# train
net.train()
for epoch in range(1, args["n_epochs"] + 1):
    running_loss = 0.0
    for i, batch in enumerate(fit_loader, 1):
        x, y = batch[0].to(device), batch[1].to(device)
        opt.zero_grad()
        y_pred = net(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        opt.step()

        running_loss += loss.item()
        if i % log_interval == log_interval - 1:
            print(
                "[{}/{}, {}/{}] running_loss: {:.3f}".format(
                    epoch,
                    args["n_epochs"],
                    i,
                    len(fit_loader),
                    running_loss / log_interval,
                )
            )
print("training finished.")

# eval
net.eval()
val_loader = data.DataLoader(
    dataset=MNIST(
        "tests/mnist_guided_opt/mnist_dataset",
        train=False,
        download=True,
        transform=transform,
    ),
    batch_size=args["batch_size"],
    shuffle=True,
)

with torch.no_grad():
    correct = 0
    total = 0
    for batch in val_loader:
        x, y = batch[0].to(device), batch[1].to(device)
        all_y_pred = net(x)
        _, y_pred = torch.max(all_y_pred.data, 1)
        total += y.size(0)
        correct += (y_pred == y).sum().item()
    print(
        "accuracy of the network on the {} test images: {}".format(
            len(val_loader.dataset), 100 * correct / total
        )
    )

    class_correct = [0] * 10
    class_total = [0] * 10
    for batch in val_loader:
        x, y = batch[0].to(device), batch[1].to(device)
        x = x.view(x.shape[0], -1)

        all_y_pred = net(x)
        _, y_pred = torch.max(all_y_pred, 1)
        c = (y_pred == y).squeeze()
        for i in range(10):
            label = y[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

    for i in range(10):
        print("accuracy of {}: {:3f}".format(i, (class_correct[i] / class_total[i])))

os.makedirs("tests/mnist_guided_opt/weights", exist_ok=True)
torch.save(
    {
        "state_dict": net.state_dict(),
        "arguments": args,
    },
    "tests/mnist_guided_opt/weights/mnist_auxnet_weights.pkl",
)
