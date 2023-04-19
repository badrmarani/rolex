import torch
from torch import nn, distributions

from .base import Decoder

class GPVAE(nn.Module):
	def __init__(
		self,
		device: torch.device,
		inp_size: int,
		emb_size: int,
		lat_size: int,
		n_inducing_pts: int,
	):
		super(GPVAE, self).__init__()

		self.gp = GP(device, inp_size, lat_size, n_inducing_pts)
		self.decoder = Decoder(lat_size, emb_size, inp_size)

	def reparameterization(self, mu, var):
		if self.training:
			std = var.exp()
			z = mu + std * torch.randn_like(std)
		else:
			z = mu
		return z

	def forward(self, tensor):
		mu, var = self.gp(tensor)
		z = self.reparameterization(mu, var)
		xhat = self.decoder(z)
		return xhat, mu, var

class GP(nn.Module):
	def __init__(
		self,
		device,
		inp_size: int,
		lat_size: int,
		n_inducing_pts: int,
	) -> None:
		super().__init__()
		self.log_scale_ = nn.Parameter(torch.tensor([0.0], device=device))

		self.lat_size = lat_size

		self.U = nn.Parameter(torch.randn(size=(n_inducing_pts, inp_size), device=device)) # ind_pts
		self.Y = nn.Parameter(torch.randn(size=(n_inducing_pts, lat_size), device=device)) 
		self.V = nn.Parameter(
			torch.ones(
				size=(n_inducing_pts, lat_size),
				device=device,
			).clamp(min=0.0, max=None)
		) # noise

	def rbf_kernel(self, A, B):
		return torch.exp(- 0.5 * torch.cdist(A, B, p=2).div(self.scale))

	@property
	def scale(self):
		return self.log_scale_.exp()

	def forward(self, x):
		kuu = self.rbf_kernel(self.U, self.U) # size(M, M)
		kxx = self.rbf_kernel(x.unsqueeze(0), x.unsqueeze(0)).squeeze(0) # size(BS, BS)
		kxu = self.rbf_kernel(
			x.unsqueeze(0),
			self.U.unsqueeze(0),
		).squeeze(0) # size(BS, M)

		mud, vard = [], []
		for i in range(self.lat_size):
			Vi = self.V[..., i] # size(M,)
			Yi = self.Y[..., i] # size(M,)
			KVi = kuu + Vi * torch.eye(Vi.size(0)).to(Vi.device) # size(M, M)
			L = torch.linalg.cholesky(KVi)
			alpha = torch.linalg.solve(
				L.t(), torch.linalg.solve(L, Yi)
			).unsqueeze(1)

			beta = torch.linalg.solve(
				L.t(), torch.linalg.solve(L, kxu.t())
			)

			mud += [kxu.mm(alpha)]
			vard += [torch.diag(kxx - kxu.mm(beta)).unsqueeze(-1)]

		mu = torch.cat(mud, dim=1)
		var = torch.cat(vard, dim=-1)
		
		return mu, var
	