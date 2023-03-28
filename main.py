import os

import torch
import pandas as pd
from src.dataset import FullDS
from src.losses import ELBOLoss
from src.models import VAE
from src.utils import train_one_epoch, train_test_split
from torch.utils import data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float

root = "data/fulldataset.csv"
df = pd.read_csv(root, sep=";")
# df.drop(columns=["Unnamed: 0", "ROW_ID"], inplace=True)
df.drop(
    columns=[
        col for col in df.columns[df.isna().any()] 
        if col.startswith("data")
    ],
    inplace=True
)

fit = FullDS(
    df,
    device = device,
)

# fit, val = train_test_split(fit, 0.8)
loader = data.DataLoader(fit, batch_size=32)

model = VAE(
    inp_size = 209,
    emb_size = 100,
    lat_size = 20,
    out_size = 209,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = ELBOLoss()


n_epochs = 10
save_intervall = n_epochs
filename = "model.pth"

for epoch in range(1, n_epochs+1):
    model = train_one_epoch(
        epoch,
        model,
        loader,
        loss_fn,
        optimizer,
        device,
        log_intervall=100,
    )

    if not epoch%save_intervall:
        torch.save({
            "epoch": epoch, 
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }, os.path.join("logs/", filename))
