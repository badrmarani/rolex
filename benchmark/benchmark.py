import logging
import torch

from dataset import JanusDataset
from nn_benchmark import Encoder, Decoder
from hydra.utils import get_class

class Benchmark():
    def __init__(self, config) -> None:
        self.config = config
        self.device = torch.device(self.config.device)

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
        try:
            del self.encoder
            del self.decoder
        except NameError:
            self.encoder = Encoder(
                data_dim=self.input_data_dim,
                compress_dims=self.config.compress_dims,
                embedding_dim=self.config.embedding_dim,
            )

            self.decoder = Decoder(
                embedding_dim=self.config.embedding_dim,
                decompress_dims=self.config.compress_dims[::-1],
                data_dim=self.input_data_dim,
            )

    def configure_model_config(self, module, root):
        model_config = get_class(module)
        model_config = model_config.from_json_file(root)
        model_config.input_dim = self.input_data_dim
        
    def run(self):
        models = self.config.models
        model_configs = self.config.model_configs.module
        for i, (module, model_config_module) in enumerate(zip(models, model_configs)):
            # root = self.config.model_configs.root[i]
            # model_config = self.configure_model_config(model_config_module, root)

            self.configure_nn()
            # model = get_class(module)(
            #     model_config=model_config,
            #     encoder=self.encoder,
            #     decoder=self.decoder,
            # )
            # print(f"\t+ Training model {model.__class__.__name__}")
            break