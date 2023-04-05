import torch
from torch import nn

class Encoder(nn.Module):
	def __init__(
		self,
		dropout: bool,
		inp_size: int,
		emb_size: int,
		lat_size: int,
	):
		super(Encoder, self).__init__()

		encode = [
			nn.Linear(inp_size, emb_size), nn.Tanh(),
			nn.Linear(emb_size, emb_size), nn.Tanh(),
		]

		if dropout:
			drop = nn.Dropout(0.2)
			encode.insert(2, drop)
			encode.insert(5, drop)
		self.encode = nn.Sequential(*encode)

		self.mu = nn.Linear(emb_size, lat_size)
		self.logvar = nn.Linear(emb_size, lat_size)

	def forward(self, tensor):
		tmp = self.encode(tensor)
		return (
			self.mu(tmp),
			self.logvar(tmp),
		)


class Decoder(nn.Module):
	def __init__(
		self,
		dropout: bool,
		lat_size: int,
		emb_size: int,
		out_size: int,
	):
		super(Decoder, self).__init__()

		decode = [
			nn.Linear(lat_size, emb_size), nn.Tanh(),
			nn.Linear(emb_size, out_size), nn.Sigmoid(),
		]

		if dropout:
			drop = nn.Dropout(0.2)
			decode.insert(2, drop)

		self.decode = nn.Sequential(*decode)

	def forward(self, tensor):
		return self.decode(tensor)
