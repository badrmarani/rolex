import torch
from torch import nn, distributions


def LDE(log_a, log_b):
    max_log = torch.max(log_a, log_b)
    min_log = torch.min(log_a, log_b)
    return max_log + torch.log(1 + torch.exp(min_log - max_log))


def log_gaussian_likelihood(xhat, x):
    # here, we assume that std == 1.0
    return - 0.5 * nn.functional.mse_loss(
        xhat,
        x,
        reduction="none",
    ).sum(dim=-1) # size(BS,)

class Encoder(nn.Module):
    def __init__(self, inp_size, emb_sizes, lat_size, add_dropouts):
        super(Encoder, self).__init__()

        seq = []
        for i, emb_size in enumerate(emb_sizes):
            if not i:
                pre_emb_size = inp_size
            seq += [
                nn.Linear(pre_emb_size, emb_size), nn.ReLU(),
            ]

            if add_dropouts:
                seq += [nn.Dropout(0.5)]            

            pre_emb_size = emb_size

        self.seq = nn.Sequential(*seq)
        self.embedding = nn.Linear(pre_emb_size, lat_size)
        self.log_covariance = nn.Linear(pre_emb_size, lat_size)
        
    def forward(self, x: torch.Tensor):
        t = self.seq(x)
        output = {}
        output["mu"] = self.embedding(t)
        output["logvar"] = self.log_covariance(t)
        return output

class Decoder(nn.Module):
    def __init__(self, lat_size, emb_sizes, add_dropouts):
        super(Decoder, self).__init__()

        emb_sizes = emb_sizes[::-1]

        seq = []
        for i, emb_size in enumerate(emb_sizes):
            if not i:
                pre_emb_size = lat_size
            seq += [nn.Linear(pre_emb_size, emb_size)]
            if i != len(emb_sizes)-1:
                seq += [nn.ReLU()]
                if add_dropouts:
                    seq += [nn.Dropout(0.5)]

            pre_emb_size = emb_size

        self.reconstruction = nn.Sequential(*seq)

    def forward(self, x: torch.Tensor):
        return self.reconstruction(x)


class GuidedVAE(nn.Module):
    def __init__(
        self,
        inp_size: int,
        emb_sizes: int,
        lat_size: int,
        is_guided: bool,
        add_dropouts: bool,
        uncertainty_threshold_value: float,
        n_gradient_steps: int,
        gradient_scale: float,
        normalize_gradient: bool,
        n_simulations: int,
        n_sampled_outcomes: int,
        auxiliary_net: nn.Module,
    ):
        super(GuidedVAE, self).__init__()

        self.encoder = Encoder(inp_size, emb_sizes, lat_size, add_dropouts)
        self.decoder = Decoder(lat_size, emb_sizes, add_dropouts)
        self.auxiliary_net = auxiliary_net

        self.is_guided = is_guided

        self.uncertainty_threshold_value = uncertainty_threshold_value
        self.n_gradient_steps = n_gradient_steps
        self.gradient_scale = gradient_scale
        self.normalize_gradient = normalize_gradient
        self.n_simulations = n_simulations
        self.n_sampled_outcomes = n_sampled_outcomes

    def rsample(self, mu, logvar):
        if self.training:
            eps = torch.randn_like(logvar, device=logvar.device)
            std = logvar.mul(0.5).exp()
            return mu + std * eps
        return mu

    def loss_function(self, x, xhat, mu, logvar):
        kld_loss = -0.5 * torch.sum(1+logvar-mu.pow(2)-logvar.exp(), dim=-1)

        bs = x.size(0)
        rec_loss = nn.functional.mse_loss(
            xhat.view(bs, -1), x.view(bs, -1),
            reduction="none",
        ).sum(dim=-1)

        return (
            (rec_loss + kld_loss).mean(dim=0),
            rec_loss.mean(dim=0),
            kld_loss.mean(dim=0),
        )

    def enable_dropout(self):
        for m in self.modules():
            if m.__class__.__name__.startswith("Dropout"):
                m.train()

    @torch.no_grad()
    def mutual_information(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        log_mi = []

        self.train()
        for s in range(self.n_simulations):
            all_log_psm = []
            
            xhat = self.decoder(z)
            
            for _ in range(self.n_sampled_outcomes):        
                self.eval(); self.enable_dropout()
                
                xxhat = self.decoder(z)
                log_psm = - nn.functional.gaussian_nll_loss(
                    xxhat,
                    xhat,
                    torch.ones_like(x, device=x.device),
                    reduction="none"
                ).sum(-1)

                all_log_psm.append(log_psm)

            all_log_psm = torch.stack(all_log_psm, dim=1)
            log_ps = - torch.log(torch.tensor(self.n_sampled_outcomes).float()) + torch.logsumexp(all_log_psm, dim=1)
            
            right_log_hs = log_ps + torch.log(-log_ps)
            psm_log_psm = all_log_psm + torch.log(-all_log_psm)
            left_log_hs = - torch.log(torch.tensor(self.n_sampled_outcomes).float()) + torch.logsumexp(psm_log_psm, dim=1)

            tmp_log_hs = LDE(left_log_hs, right_log_hs) - log_ps
            log_mi.append(tmp_log_hs)

            if not s%10:
                print("[DEBUG] >>>> mutual info. [{}/{}]".format(
                    s, self.n_simulations, 
                ), end=" ")

        log_mi = torch.stack(log_mi, dim=1)
        log_mi_avg = - torch.log(torch.tensor(self.n_simulations).float()) + torch.logsumexp(log_mi, dim=1)
        
        return log_mi_avg.exp()

    def gradient_ascent_optimisation(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        
        for step in range(self.n_gradient_steps):
            P = self.auxiliary_net(z)
            gradient = torch.autograd.grad(
                outputs=P,
                inputs=z,
                grad_outputs=torch.ones_like(P, device=z.device),
                retain_graph=False,
            )[0]

            if self.normalize_gradient:
                gradient /= gradient.norm(2)

            updated_z = z + gradient*self.gradient_scale
            mi = self.mutual_information(x, z)

            mask = (mi <= self.uncertainty_threshold_value)
            mask = mask.unsqueeze(-1).repeat(1, 2)
            updated_z = torch.where(mask, updated_z, z)

            if not step%10:
                print("[DEBUG] gradient ascent opt. [{}/{}]".format(
                    step, self.n_gradient_steps
                ))

        return updated_z

    def forward(self, x:torch.Tensor):
        t = self.encoder(x)
        mu, logvar = t.values()
        z = self.rsample(mu, logvar)

        if self.is_guided:
            z = self.gradient_ascent_optimisation(x, z)

        xhat = self.decoder(z)
        return xhat, mu, logvar

