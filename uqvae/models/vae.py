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
		return mu + std * eps

	def loss_function(self, x, xhat, mu, logvar):
		bs = x.size(0)
		rec_loss = nn.functional.mse_loss(
			xhat.view(bs, -1),
			x.view(bs, -1),
			reduction="none",
		).sum(dim=-1)

		kld_loss = -0.5 * torch.sum(1+logvar-mu.pow(2)-logvar.exp(), dim=-1)

		return (
			(rec_loss + kld_loss).mean(dim=0),
			rec_loss.mean(dim=0).detach(),
			kld_loss.mean(dim=0).detach(),
		)

	def forward(self, tensor):
		mu, logvar = self.encoder(tensor)
		z = self.reparameterization(mu, logvar)
		xhat = self.decoder(z)
		return xhat, mu, logvar
