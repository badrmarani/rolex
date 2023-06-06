from torch import nn


class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        qzx = self.encoder(x)
        z = qzx.rsample()
        pxz = self.decoder(z)
        return qzx, pxz
