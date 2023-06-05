import random
from contextlib import contextmanager

import numpy as np
import torch


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
