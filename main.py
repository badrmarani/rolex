import os

import torch
import pandas as pd
from src.dataset import FullDS
from src.losses import ELBOLoss
from src.models import VAE
from src.utils import train_one_epoch, train_test_split, plot_latent_space
from torch.utils import data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float


os.makedirs("logs/", exist_ok=True)
os.makedirs(log_weight := "logs/weights/", exist_ok=True)
os.makedirs(log_figure := "logs/figures/", exist_ok=True)


root = "data/fulldataset.csv"
df = pd.read_csv(root, sep=";")
# df.drop(columns=["Unnamed: 0", "ROW_ID"], inplace=True)
df.drop(
    columns=[col for col in df.columns[df.isna().any()] if col.startswith("data")],
    inplace=True,
)

fit = FullDS(
    df,
    device=device,
)

fit, val = train_test_split(fit, 0.8)
loader = data.DataLoader(fit, batch_size=128, shuffle=True)

batch_size = 32
inp_size = 209
emb_size = 100
lat_size = 2
out_size = 209

model = VAE(
    inp_size=inp_size,
    emb_size=emb_size,
    lat_size=lat_size,
    out_size=out_size,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = ELBOLoss()


n_epochs = 100
save_intervall = 10
filename = "{}_Ep{}_Emb{}_Lat{}.pth".format(
    model.__class__.__name__,
    n_epochs,
    emb_size,
    lat_size,
)


for epoch in range(1, n_epochs + 1):
    model = train_one_epoch(
        epoch,
        model,
        loader,
        loss_fn,
        optimizer,
        device,
        log_intervall=len(loader) + 1,
    )

    if not epoch % save_intervall:
        plot_latent_space(
            epoch=epoch,
            model=model,
            df=df,
            device=device,
            n_samples=5000,
            save=os.path.join(
                log_figure,
                "{}_Ep{}_Emb{}_Lat{}.png".format(
                    model.__class__.__name__,
                    epoch,
                    emb_size,
                    lat_size,
                ),
            ),
        )

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            os.path.join(log_weight, filename),
        )
