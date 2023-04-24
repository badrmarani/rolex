import datetime
import decimal
import gc
import importlib
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import yaml
from pythae import models
from pythae.data.datasets import BaseDataset
from pythae.pipelines import TrainingPipeline
from pythae.trainers import BaseTrainer, BaseTrainerConfig
from ray import air, tune
from ray.tune.schedulers import ASHAScheduler
from sklearn.model_selection import train_test_split
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import MNIST
from utils import *
from base_model import Encoder, Decoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = get_dataset(args["dataset_filename"], test_size=0.10, random_state=42)
train_data = data["train"]["torch"]
test_data = data["test"]["torch"]
inp_size = train_data.data.size(1)
emb_sizes = [inp_size//i for i in range(1, 4)]
lat_size = args["lat_size"]


save_models_dirname = os.path.join(dirname, "all_models_logs")
save_tensorboard_dirname = os.path.join(save_models_dirname, "tensorboard_"+training_signature)
os.makedirs(save_models_dirname, exist_ok=True)
os.makedirs(save_tensorboard_dirname, exist_ok=True)

# if os.path.exists(save_models_dirname):
#     import shutil; shutil.rmtree(save_models_dirname) # DEBUG

with open(os.path.join(dirname, "models.txt"), "r") as f:
    all_models = f.read().splitlines()

module = importlib.import_module("pythae.models")
for name, config in zip(all_models[::2], all_models[1::2]):
    writer = SummaryWriter(log_dir=os.path.join(
        save_tensorboard_dirname, name
    ))

    print("="*20, "Training {}...".format(name), "="*20)
    callbacks = [TBLogger(writer)]
    my_model, my_config = (
        getattr(module, name),
        getattr(module, config)
    )

    model_config = my_config(
        input_dim = (inp_size,),
        latent_dim = lat_size,
        uses_default_encoder = False,
        uses_default_decoder = False,
    )
    
    encoder_type = "base"
    if name.lower() == "svae":
        encoder_type = name

    encoder = Encoder(inp_size, emb_sizes, lat_size, encoder_type).to(device)
    decoder = Decoder(lat_size, emb_sizes, inp_size).to(device)

    model = my_model(
        model_config = model_config,
        encoder = encoder,
        decoder = decoder,
    )

    training_config = get_configs(name)    
    trainer = BaseTrainer(
        model = model,
        train_dataset = train_data,
        eval_dataset = test_data,
        training_config = training_config,
        callbacks = callbacks,
        repeat_number=args["n_repeats"],
        training_signature=training_signature,
    )

    trainer.train()
    # save_quality_report(model, data["eval"]["pandas"], name)
    writer.close()

    del model
    del trainer
    del training_config
    del encoder
    del decoder
    with torch.no_grad():
        torch.cuda.empty_cache()
    gc.collect()
