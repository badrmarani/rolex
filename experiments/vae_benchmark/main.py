import torch

from torch.utils.tensorboard import SummaryWriter

import importlib
import datetime
import os

from utils import Encoder, Decoder, TBLogger

from pythae import models
from pythae.pipelines import TrainingPipeline
from pythae.trainers import (
    BaseTrainerConfig,
    AdversarialTrainerConfig,
    CoupledOptimizerAdversarialTrainerConfig,
    CoupledOptimizerTrainerConfig,
)

inp_size = 20
emb_sizes = [
    inp_size//i
    for i in range(1, 4)
]

lat_size = 2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

XX = torch.randn(size=(6000, inp_size)).to(device)

decoder = Decoder(lat_size, emb_sizes, inp_size).to(device)

training_signature = str(datetime.datetime.now())[0:19].replace(" ", "_").replace(":", "-")
output_dir = "experiments/vae_benchmark/all_models_logs"
ts_output_dir = os.path.join(output_dir, "tensorboard_"+training_signature)
os.makedirs(ts_output_dir, exist_ok=True)

training_base_config = BaseTrainerConfig(
    no_cuda = False,
    num_epochs = 40,
    keep_best_on_train = True,
    output_dir = output_dir,
    per_device_train_batch_size = 64,
    per_device_eval_batch_size = 64,
    optimizer_cls = "Adam",
    learning_rate = 1e-4,
    optimizer_params = {
        "betas": (0.91, 0.995),
        "weight_decay": 0.05,
    },
    scheduler_cls="ReduceLROnPlateau",
    scheduler_params={"patience": 5, "factor": 0.5}
)


training_aae_config = AdversarialTrainerConfig(
    no_cuda = False,
    num_epochs = 1,
    keep_best_on_train = True,
    output_dir = output_dir,
    per_device_train_batch_size = 64,
    per_device_eval_batch_size = 64,
    autoencoder_optimizer_cls = "Adam",
    autoencoder_learning_rate = 1e-4,
    autoencoder_optimizer_params = {
        "betas": (0.91, 0.995),
        "weight_decay": 0.05,
    },
    autoencoder_scheduler_cls="ReduceLROnPlateau",
    autoencoder_scheduler_params={"patience": 5, "factor": 0.5},
    discriminator_optimizer_cls = "Adam",
    discriminator_learning_rate = 1e-4,
    discriminator_optimizer_params = {
        "betas": (0.91, 0.995),
        "weight_decay": 0.05,
    },
    discriminator_scheduler_cls="ReduceLROnPlateau",
    discriminator_scheduler_params={"patience": 5, "factor": 0.5}
)

training_vaegan_config = CoupledOptimizerAdversarialTrainerConfig(
    no_cuda = False,
    num_epochs = 1,
    keep_best_on_train = True,
    output_dir = output_dir,
    per_device_train_batch_size = 64,
    per_device_eval_batch_size = 64,
    encoder_optimizer_cls = "Adam",
    encoder_learning_rate = 1e-4,
    encoder_optimizer_params = {
        "betas": (0.91, 0.995),
        "weight_decay": 0.05,
    },
    encoder_scheduler_cls="ReduceLROnPlateau",
    encoder_scheduler_params={"patience": 5, "factor": 0.5},
    decoder_optimizer_cls = "Adam",
    decoder_learning_rate = 1e-4,
    decoder_optimizer_params = {
        "betas": (0.91, 0.995),
        "weight_decay": 0.05,
    },
    decoder_scheduler_cls="ReduceLROnPlateau",
    decoder_scheduler_params={"patience": 5, "factor": 0.5},
    discriminator_optimizer_cls = "Adam",
    discriminator_learning_rate = 1e-4,
    discriminator_optimizer_params = {
        "betas": (0.91, 0.995),
        "weight_decay": 0.05,
    },
    discriminator_scheduler_cls="ReduceLROnPlateau",
    discriminator_scheduler_params={"patience": 5, "factor": 0.5}
)

training_piwae_config = CoupledOptimizerTrainerConfig(
    no_cuda=False,
    num_epochs=1,
    keep_best_on_train=True,
    output_dir=output_dir,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    encoder_optimizer_cls="Adam",
    encoder_learning_rate=1e-4,
    encoder_optimizer_params={
        "betas": (0.91, 0.995),
        "weight_decay": 0.05,
    },
    encoder_scheduler_cls="ReduceLROnPlateau",
    encoder_scheduler_params={"patience": 5, "factor": 0.5},
    decoder_optimizer_cls="Adam",
    decoder_learning_rate=1e-4,
    decoder_optimizer_params={
        "betas": (0.91, 0.995),
        "weight_decay": 0.05,
    },
    decoder_scheduler_cls="ReduceLROnPlateau",
    decoder_scheduler_params={"patience": 5, "factor": 0.5},
)

with open("experiments/vae_benchmark/models.txt", "r") as f:
    all_models = f.read().splitlines()

module = importlib.import_module("pythae.models")

for name, config in zip(all_models[::2], all_models[1::2]):

    print("Training {}...".format(name))

    Model, Config = (
        getattr(module, name),
        getattr(module, config)
    )

    model_config = Config(
        input_dim = (inp_size,),
        latent_dim = lat_size,
        uses_default_encoder = False,
        uses_default_decoder = False,
    )
    
    mtype = "base"
    if name.lower() == "svae":
        mtype = name
    encoder = Encoder(inp_size, emb_sizes, lat_size, mtype).to(device)

    model = Model(
        model_config = model_config,
        encoder = encoder,
        decoder = decoder,
    )

    if name.lower() in ["adversarial_ae", "factorvae"]:
        training_config = training_aae_config
    elif name.lower() == "vaegan":
        training_config = training_vaegan_config
    elif name.lower() == "piwae":
        training_config = training_piwae_config
    else:
        training_config = training_base_config
    
    pipeline = TrainingPipeline(
    	training_config = training_config,
    	model = model
    )

    writer = SummaryWriter(log_dir=os.path.join(ts_output_dir, name))
    callbacks = [TBLogger(writer),]

    pipeline(
        train_data = XX,
        eval_data = XX,
        callbacks = callbacks,
    )

    writer.close()