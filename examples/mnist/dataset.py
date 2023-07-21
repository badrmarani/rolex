import torch
from torchvision.datasets import MNIST

from rolex.dataset import DataModule


class MNISTDataModule(DataModule):
    def __init__(self, config, data_weighter=None, valid=False) -> None:
        super().__init__(config, data_weighter, valid)

    def prepare(self, train, root):
        mnist = MNIST(train=train, root=root, download=True)
        indices = torch.where(mnist.targets == self.config.digit, 1.0, 0.0).nonzero()
        data = mnist.data[indices, ...].flatten(1)

        # data = (data-data.min())/(data.max()-data.min())
        data = data.div(255.0)

        targets = torch.sum(data, dim=-1)
        if self.config.normalize_targets:
            m, M = torch.min(targets), torch.max(targets)
            targets = (targets - m) / (M - m)
        return data, targets
