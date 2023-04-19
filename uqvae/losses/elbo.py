import torch
from torch import nn, distributions

class ELBOLoss(nn.Module):
	def __init__(self, device: torch.device) -> None:
		super().__init__()

		self.log_scale = nn.Parameter(
			torch.tensor([0.0], device=device),
			requires_grad=True,
		)
	
	# def kl_divergence(self, z, mu, std):
	# 	log_pz = distributions.Normal(
	# 		torch.zeros_like(mu),
	# 		torch.ones_like(std),
	# 	).log_prob(z)
		
	# 	log_qzx = distributions.Normal(mu, std).log_prob(z)

	# 	kld_loss = log_qzx - log_pz
	# 	return kld_loss.sum(-1)

	def kl_divergence(self, p, q=None):
		if q is None:
			q = distributions.Normal(0.0, 1.0)

		return distributions.kl_divergence(p, q).sum(dim=-1)

	# def kl_divergence(self, z, q_mu, q_std, p_mu=0.0, p_std=1.0):
	# 	log_q = distributions.MultivariateNormal(
	# 		q_mu, q_std.diag_embed()
	# 	).log_prob(z)

	# 	if isinstance(p_mu, float) and isinstance(p_std, float):
	# 		p_mu = torch.tensor(
	# 			[p_mu],
	# 			device=z.device,
	# 			dtype=z.dtype
	# 		).repeat(z.size())
	# 		p_std = torch.tensor(
	# 			[p_std],
	# 			device=z.device,
	# 			dtype=z.dtype
	# 		).repeat(z.size()).diag_embed()
	# 		log_p = distributions.MultivariateNormal(
	# 			p_mu, p_std
	# 		).log_prob(z)
	# 	else:
	# 		# in case the prior p(z) is not gaussian
	# 		raise NotImplementedError
	# 	kld = log_q - log_p
	# 	return kld.sum(dim=-1)

	def gaussian_likelihood(self, xhat, x):
		scale = self.log_scale.exp()
		log_pxz = distributions.Normal(xhat, scale).log_prob(x)
		return log_pxz.sum(-1)

	def forward(self, x, xhat, z, mu, logvar):
		std = logvar.mul(0.5).exp()

		q = distributions.Normal(mu, std)
		kld_loss = self.kl_divergence(q)
		
		rec_loss = - self.gaussian_likelihood(xhat, x)
		elbo_loss = kld_loss + rec_loss
		elbo_loss = elbo_loss.mean(dim=0)
		
		return (
			elbo_loss,
			rec_loss.mean(dim=0),
			kld_loss.mean(dim=0),
		)
