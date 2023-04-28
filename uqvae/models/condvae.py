import torch
from torch import nn

from .base import Encoder, Decoder

class CondVAE(nn.Module):
	def __init__(
		self,
		data_dim,
		compress_dims,
		decompress_dims,
		embedding_dim,
		add_dropouts=False,
		p=None,
	):
		super(CondVAE, self).__init__()
		self.cond_encoder = CondEncoder(data_dim, compress_dims, embedding_dim, add_dropouts, p)
		self.decoder = Decoder(embedding_dim, decompress_dims, data_dim)
		self.fc = nn.Linear(1, embedding_dim)

	def rsample(self, mu, logvar):
		std = logvar.mul(0.5).exp()
		z = mu + std * torch.randn_like(std)
		return z

	def forward(self, x):
		mu_z, logvar_z, mu_y, logvar_y = self.encoder(x)
		
		z_given_x = self.rsample(mu_z, logvar_z) # z ~ q(z|x)
		y = self.rsample(mu_y, logvar_y)
		z_given_y = self.fc(y) # z ~ q(z|y) = N(zy, 1)
		recon_x, sigma = self.decoder(z_given_x)
  
		return recon_x, mu_z, logvar_z, mu_y, logvar_y, z_given_x, z_given_y, sigma


class CondEncoder(nn.Module):
	def __init__(
		self,
		data_dim,
		compress_dims,
		embedding_dim,
		add_dropouts=False,
		p=None,
	):

		super().__init__()

		self.encoder = Encoder(
			data_dim,
			compress_dims,
			embedding_dim,
			add_dropouts,
			p,
			is_conditioned=True,
		)
		
		self.fc1_z = nn.Linear(compress_dims[-1], embedding_dim)
		self.fc2_z = nn.Linear(compress_dims[-1], embedding_dim)
		self.fc1_y = nn.Linear(compress_dims[-1], 1)
		self.fc2_y = nn.Linear(compress_dims[-1], 1)

	def forward(self, x):
		x = self.encoder(x)
		return (
			self.fc1_z(x), self.fc1_z(x),
			self.fc2_y(x), self.fc2_y(x),
		)
