from torch.nn import Module

from .encoders import *
from .common import *
from .diffusion import *


class GaussianVAE(Module):
    def __init__(self, args):
        super().__init__()
        self.z = None
        self.z_log_var = None
        self.loss_reverse = None
        self.encoder = PointNetEncoder(args.latent_dim)
        self.diffusion = DiffusionPoint(z_dim=args.latent_dim)

    def forward(self, x):
        z_mean, self.z_log_var = self.encoder(x)
        self.z = reparameterize_gaussian(z_mean, self.z_log_var)
        self.loss_reverse = self.diffusion(x, self.z)

    def get_loss(self, kl_weight=1.0):
        # get KL by diffusion
        log_pz = standard_normal_logprob(self.z).sum(dim=1)
        entropy = gaussian_entropy(log_var=self.z_log_var)
        loss_prior = (-log_pz - entropy).mean()
        loss = kl_weight * loss_prior + self.loss_reverse
        # record entropy/log_pz/loss_reverse
        return loss

    def sample(self):
        pass
