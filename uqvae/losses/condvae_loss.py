import torch
from torch import nn, distributions


class CondELBOLoss(nn.Module):
	def __init__(self):
		super().__init__()

    def nll_gaussian(self, x, x_recon)
        pass

	def gaussian_likelihood(
		self,
		input,
		target,
		scale = None,
	):
		if scale is None:
			scale = self.log_scale.exp()
		
		log_pxz = distributions.Normal(
			input, scale
		).log_prob(target)
		return log_pxz.sum(-1)
	
	def kl_divergence(
		self, z, mu_z, std_z, zy,
	):
		log_qzx = distributions.Normal(
			mu_z, std_z
		).log_prob(z)
		log_qzy = distributions.Normal(
			zy,
			torch.ones_like(zy),
		).log_prob(z)
		kld = log_qzx - log_qzy
		return kld.sum(-1)
		
	def forward(
		self, x, y, xhat, z, mu_z, std_z, zy, mu_y, std_y  
	):
		rec_loss = self.gaussian_likelihood(xhat, x)
		label_loss = self.gaussian_likelihood(mu_y, y, scale=std_y)
		kld_loss = self.kl_divergence(z, mu_z, std_z, zy)
		loss = rec_loss + label_loss - kld_loss
		return - loss.mean()