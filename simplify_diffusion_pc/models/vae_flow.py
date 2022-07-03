import torch
from torch.nn import Module

from .encoders import *
from .common import *
from .diffusion import *
from .flow import *


class FlowVAE(Module):
    def __init__(self, args):
        super().__init__()
        self.z = None
        self.w = None
        self.delta_log_pw = None
        self.z_log_var = None
        self.loss_reverse = None
        self.encoder = PointNetEncoder(args.latent_dim)
        self.flow = build_latent_flow(args)
        self.diffusion = DiffusionPoint(z_dim=args.latent_dim, device=args.device)

    def forward(self, x):
        z_mean, self.z_log_var = self.encoder(x)
        self.z = reparameterize_gaussian(z_mean, self.z_log_var)
        self.w, self.delta_log_pw = self.flow(self.z, torch.zeros([x.size(0), 1]).to(self.z), reverse=False)
        self.loss_reverse = self.diffusion(x, self.z)
        return self.z, self.loss_reverse

    def get_loss(self, kl_weight=0.001):
        # TODO: why the expectation did not calculate mean?
        # (B,1) Monte Carlo method: (1/n)*sum(f(X)) = Ep[f(x)]
        log_pw = standard_normal_logprob(self.w).sum(dim=1, keepdim=True)
        log_pz = log_pw - self.delta_log_pw
        # (B,1), H(q(z|x0)): entropy of multivariate normal distribution
        entropy = gaussian_entropy(log_var=self.z_log_var)

        loss_prior = (-log_pz - entropy).mean()
        loss = kl_weight * loss_prior + self.loss_reverse
        # record entropy/log_pz/loss_reverse
        return loss

    def sample(self, w, num_points, truncate_std=None):
        if truncate_std is not None:
            w = truncated_normal_(w, mean=0, std=1, trunc_std=truncate_std)
        z_context = self.flow(w, reverse=True).view(w.size(0), -1)
        samples = self.diffusion.reverse_sample(num_points, z_context)
        return samples
