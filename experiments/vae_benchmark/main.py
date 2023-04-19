import datetime
import decimal
import gc
import importlib
import os
import warnings

import pandas as pd
import torch
import yaml
from pythae import models
from pythae.data.datasets import BaseDataset
from pythae.pipelines import TrainingPipeline
from pythae.trainers import BaseTrainer, BaseTrainerConfig
from ray import air, tune
from ray.tune.schedulers import ASHAScheduler
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import MNIST

from utils import *

warnings.filterwarnings("ignore")

def train_ray(model_configs):
    Model, Config = (
        getattr(module, name),
        getattr(module, config)
    )

    if model_configs is None:
        model_configs = {}

    model_config = Config(
        input_dim = (inp_size,),
        latent_dim = lat_size,
        uses_default_encoder = False,
        uses_default_decoder = False,
        **model_configs,
    )
    
    mtype = "base"
    if name.lower() == "svae":
        mtype = name

    ENCODER = Encoder(inp_size, emb_sizes, lat_size, mtype).to(device)
    DECODER = Decoder(lat_size, emb_sizes, inp_size).to(device)

    model = Model(
        model_config = model_config,
        encoder = ENCODER,
        decoder = DECODER,
    )

    if name.lower() in ["adversarial_ae", "factorvae"]:
        training_config = CONFIGS["adversarial_trainer_config"]
    elif name.lower() == "vaegan":
        training_config = CONFIGS["coupled_optimizer_adversarial_trainer_config"]
    elif name.lower() == "piwae":
        training_config = CONFIGS["coupled_optimizer_trainer_config"]
    else:
        training_config = CONFIGS["training_base_config"]
    
    trainer = BaseTrainer(
        model = model,
        train_dataset = fit_dataset,
        eval_dataset = val_dataset,
        training_config = training_config,
        callbacks = callbacks,
        repeat_number=rn,
        training_signature=training_signature,
    )

    trainer.train()
    # writer.close()

    del model
    del trainer
    del training_config
    del ENCODER
    del DECODER
    with torch.no_grad():
        torch.cuda.empty_cache()
    gc.collect()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.0,), (1.0,)),
    transforms.Lambda(torch.flatten),
])

fit = MNIST("tests/mnist/mnist/", train=True, download=True, transform=transform)
val = MNIST("tests/mnist/mnist/", train=True, download=True, transform=transform)
fit_dataset = BaseDataset(fit.data[:1000].flatten(1) / 255., torch.ones((fit.data[:1000].size(0))))
val_dataset = BaseDataset(val.data[:1000].flatten(1) / 255., torch.ones((val.data[:1000].size(0))))

inp_size = 28*28
lat_size = 2
emb_sizes = [inp_size//i for i in range(1, 4)]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_REPEATS = 10

training_signature = str(datetime.datetime.now())[0:19].replace(" ", "_").replace(":", "-")
path_name = "experiments/vae_benchmark/all_models_logs"

if os.path.exists(path_name):
    import shutil; shutil.rmtree(path_name) # DEBUG


# output_dir = os.path.join(path_name, "tensorboard_"+training_signature)
# os.makedirs(output_dir, exist_ok=True)

with open("experiments/vae_benchmark/models.txt", "r") as f:
    all_models = f.read().splitlines()

module = importlib.import_module("pythae.models")
for rn in range(1, N_REPEATS+1):

    print("="*20, "REPEAT {}/{}".format(rn, N_REPEATS), "="*20)
    for name, config in zip(all_models[::2], all_models[1::2]):
        # writer = SummaryWriter(log_dir=os.path.join(output_dir, name))
        callbacks = [TBLogger()]

        print("Training {}...".format(name))
        with open("experiments/vae_benchmark/configs/{}.yml".format(name.lower()), "r") as stream:
            search_space = yaml.safe_load(stream)

        if search_space is not None:
            print(">>> Tuning {}'s hyperparameters...".format(name))
            for x in search_space:
                search_space[x] = tune.uniform(*search_space[x])
            
            tuner = tune.Tuner(
                train_ray,
                tune_config=tune.TuneConfig(
                    num_samples=100,
                    mode="min",
                    metric="eval_epoch_loss",
                    # scheduler=ASHAScheduler(stop_last_trials=False),
                ),
                param_space=search_space,
            )

            results = tuner.fit()
            for i in range(len(results)): 
                result = results[i].metrics
                d = dict()
                d["eval_epoch_loss"] = result["eval_epoch_loss"]
                for x in result["config"]:
                    d[x] = result["config"][x]
                
                if not i:
                    df = pd.DataFrame(columns=d.keys())

                df.loc[-1] = d
                df.index += 1

            df.to_csv(os.path.join(
                path_name, f"{name}_rep_{rn}_results_{training_signature}.csv"
            ), sep=";", float_format="%.8f")

        else:
            continue
            callbacks = None
            train_ray(search_space)

    break