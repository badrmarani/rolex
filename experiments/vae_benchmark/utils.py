import torch
from torch import nn

from pythae.models.nn import BaseEncoder, BaseDecoder
from pythae.models.base.base_utils import ModelOutput
from pythae.trainers.training_callbacks import MetricConsolePrinterCallback
from pythae.trainers.base_trainer.base_training_config import BaseTrainerConfig

class TBLogger(MetricConsolePrinterCallback):
    def __init__(self, writer):
        self.writer = writer
        self.current_indx = 0

    def on_log(self, training_config: BaseTrainerConfig, logs, **kwargs):
        epoch_train_loss = logs.get("train_epoch_loss", None)
        epoch_eval_loss = logs.get("eval_epoch_loss", None)

        self.current_indx += 1
        self.writer.add_scalar("loss/eval", epoch_eval_loss, self.current_indx)
        self.writer.add_scalar("loss/train", epoch_train_loss, self.current_indx)

class Encoder(BaseEncoder):
    def __init__(self, inp_size, emb_sizes, lat_size, mtype="base"):
        BaseEncoder.__init__(self)

        seq = []
        for i, emb_size in enumerate(emb_sizes):
            if not i:
                seq += [
                    nn.Linear(inp_size, emb_size), nn.ReLU(),
                ]
            else:
                seq += [
                    nn.Linear(pre_emb_size, emb_size), nn.ReLU(),
                ]
            pre_emb_size = emb_size
        

        self.seq = nn.Sequential(*seq)

        self.embedding = nn.Linear(pre_emb_size, lat_size)

        self.mtype = mtype
        if self.mtype.lower() == "svae":
            tout = 1
        else:
            tout = lat_size
        
        self.log_covariance = nn.Linear(pre_emb_size, tout)
        
    def forward(self, x: torch.Tensor) -> ModelOutput:
        output = ModelOutput()

        emb = self.seq(x)

        output["embedding"] = self.embedding(emb)
        if self.mtype.lower() == "svae":            
            output["log_concentration"] = self.log_covariance(emb)
        else:
            output["log_covariance"] = self.log_covariance(emb)
        return output


class Decoder(BaseDecoder):
    def __init__(self, lat_size, emb_sizes, out_size):
        BaseDecoder.__init__(self)

        emb_sizes = emb_sizes[::-1]

        seq = []
        for i, emb_size in enumerate(emb_sizes):
            if not i:
                seq += [
                    nn.Linear(lat_size, emb_size), nn.ReLU(),
                ]
            else:
                seq += [nn.Linear(pre_emb_size, emb_size)]
                if i != len(emb_sizes)-1:
                    seq += [nn.ReLU()]

            pre_emb_size = emb_size
        
        self.reconstruction = nn.Sequential(*seq)
        
        
    def forward(self, x: torch.Tensor) -> ModelOutput:
        output = ModelOutput()

        output["reconstruction"] = self.reconstruction(x)
        return output