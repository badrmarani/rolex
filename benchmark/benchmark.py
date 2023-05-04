import logging
import torch

from src.pythae.pipelines import TrainingPipeline

from dataset import JanusDataset
from nn_benchmark import Encoder, Decoder

from hydra.utils import get_class

import json

def create_training_config_file(model_name, config):
    if model_name.lower() == "factorvae":
        pass
    else:
        out = {
            "name": "BaseTrainerConfig",
            "output_dir": model_name,
            "batch_size": config.batch_size,
            "num_epochs": config.n_epochs,
            "learning_rate": config.lr,
            "steps_saving": None,
            "steps_predict": config.steps_predict,
            "no_cuda": config.no_cuda,
        }
    with open("configs/base_training_config.json", "w") as f:
        json.dump(out, f)


class Benchmark():
    def __init__(self, config) -> None:
        self.config = config
        self.device = torch.device(
            "cuda" 
            if not self.config.no_cuda
            else "cpu"
        )

    def load_data(self):
        print(f"\t+ Loading {self.config.dataset.name} data...")
        dataset = JanusDataset(
            filename=self.config.dataset.root,
            n_classes_allowed=self.config.dataset.n_classes_allowed,
            device=self.device,
        )

        self.train, self.val, self.test = dataset.split(test_size=0.2, random_state=42)
        print(f"\t+ Successfully loaded {self.config.dataset.name} data")
        
        self.input_data_dim = (dataset.n_features,)

    def configure_nn(self):

        if (
            "encoder" in self.__dict__.keys() or
            "decoder" in self.__dict__.keys()
        ):
            del self.encoder
            del self.decoder

        self.encoder = Encoder(
            data_dim=self.input_data_dim[0],
            compress_dims=self.config.compress_dims,
            embedding_dim=self.config.embedding_dim,
        )

        self.decoder = Decoder(
            embedding_dim=self.config.embedding_dim,
            decompress_dims=self.config.compress_dims[::-1],
            data_dim=self.input_data_dim[0],
        )

    def configure_model_config(self, module, root):
        model_config = get_class(module).from_json_file(root)
        model_config.input_dim = self.input_data_dim
        return model_config

    def configure_training_config(self, model_name):
        training_config_modules = self.config.training_configs.module
        training_config_root = self.config.training_configs.root
        if model_name.lower() == "factorvae":
            training_config = get_class(training_config_modules[-1]).from_json_file(training_config_root[-1])
        else:
            training_config = get_class(training_config_modules[0]).from_json_file(training_config_root[0])
        return training_config

    def run(self):
        models = self.config.models
        model_configs = self.config.model_configs.module
        for i, (module, model_config_module) in enumerate(zip(models, model_configs)):
            model = get_class(module)
            create_training_config_file(model.__name__, self.config)
            for j in range(10):
                print(f"\t+ Training model {model.__name__} on configuration {j+1}")
                root = self.config.model_configs.root[i]
                model_config = self.configure_model_config(model_config_module, root)
                self.configure_nn()
                model = get_class(module)(
                    model_config=model_config,
                    encoder=self.encoder,
                    decoder=self.decoder,
                )
            
                training_config = self.configure_training_config(model.model_name)
                pipeline = TrainingPipeline(training_config=training_config, model=model)

                pipeline(train_data=(self.train,), eval_data=(self.val,))

                if not j and model.__class__.__name__ == "VAE":
                    break
                model = get_class(module)
