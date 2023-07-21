from torch import nn


class DropConnect(nn.Dropout):
    def __init__(self, p: float):
        """
        Implements DropConnect.
        Ref: https://pubmed.ncbi.nlm.nih.gov/33750847/
        """
        super(DropConnect, self).__init__()
        self.p = p

    def forward(self, input):
        return nn.functional.dropout(
            input,
            self.p,
            self.training,
            self.inplace
        ) * (1 - self.p)
