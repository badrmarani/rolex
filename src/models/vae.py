import torch
from torch import nn, distributions

from .base import Encoder, Decoder

class VAE(nn.Module):
	def __init__(
		self,
		dropout: bool,
		inp_size: int,
		emb_size: int,
		lat_size: int,
	):
		super(VAE, self).__init__()

		self.encoder = Encoder(dropout, inp_size, emb_size, lat_size).to(torch.float64)
		self.decoder = Decoder(dropout, lat_size, emb_size, inp_size).to(torch.float64)

	def reparameterization(self, mu, logvar):
		std = logvar.mul(0.5).exp()
		eps = torch.randn_like(std, dtype=mu.dtype, device=mu.device)		
		z = mu + std * eps
		return z

	def forward(self, tensor):
		mu, logvar = self.encoder(tensor)
		z = self.reparameterization(mu, logvar)
		xhat = self.decoder(z)
		return xhat, mu, logvar
