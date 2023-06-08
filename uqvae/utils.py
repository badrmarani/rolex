import random
from ast import literal_eval
from contextlib import contextmanager

import numpy as np
import torch


def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()


def lde(log_a, log_b):
    max_log = torch.max(log_a, log_b)
    min_log = torch.min(log_a, log_b)
    return max_log + torch.log(1 + torch.exp(min_log - max_log))


def add_default_trainer_args(parser, default_root=None):
    pl_trainer_grp = parser.add_argument_group("pl trainer")
    pl_trainer_grp.add_argument("--cuda", default=False, action="store_true")
    pl_trainer_grp.add_argument("--seed", type=int, default=42)
    pl_trainer_grp.add_argument("--root_dir", type=str, default=default_root)
    pl_trainer_grp.add_argument("--load_from_checkpoint", type=str, default=None)
    pl_trainer_grp.add_argument("--max_epochs", type=int, default=1000)


@contextmanager
def reproduce(seed=42):
    random.seed(seed)
    np.random.default_rng(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True
    yield


def parse_list(raw: str):
    pattern = raw.replace('"', "").replace("\\'", "'")
    return literal_eval(pattern)
