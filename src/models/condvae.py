import torch
from torch import nn

from .base import Decoder

class CondVAE(nn.Module):
	def __init__(
		self,
		inp_size: int,
		emb_size: int,
		lat_size: int,
	):
		super(CondVAE, self).__init__()     
		self.encoder = CondEncoder(inp_size, emb_size, lat_size)
		self.decoder = Decoder(lat_size, emb_size, inp_size)
		self.fc = nn.Linear(1, lat_size)

	def reparameterization(self, mu, logvar):
		std = logvar.mul(0.5).exp()
		z = mu + std * torch.randn_like(std)
		return z

	def forward(self, tensor):
		mu_z, logvar_z, mu_y, logvar_y = self.encoder(tensor)
		
		z = self.reparameterization(mu_z, logvar_z) # z ~ q(z|x)
		y = self.reparameterization(mu_y, logvar_y)
		zy = self.fc(y) # z ~ q(z|y) = N(zy, 1)
		xhat = self.decoder(z)
  
		return xhat, mu_z, logvar_z, mu_y, logvar_y, z, zy


class CondEncoder(nn.Module):
	def __init__(
		self,
		inp_size: int,
		emb_size: int,
		lat_size: int,
	):
		super(CondEncoder, self).__init__()
		self.encode = nn.Sequential(
			nn.Linear(inp_size, emb_size), nn.Tanh(),
			nn.Linear(emb_size, emb_size), nn.Tanh(),
		)

		self.mu_z = nn.Linear(emb_size, lat_size)
		self.logvar_z = nn.Linear(emb_size, lat_size)

		self.mu_y = nn.Linear(emb_size, 1)
		self.logvar_y = nn.Linear(emb_size, 1)

	def forward(self, tensor):
		tmp = self.encode(tensor)
		return (
			self.mu_z(tmp),
			self.logvar_z(tmp),
			self.mu_y(tmp),
			self.logvar_y(tmp),
		)