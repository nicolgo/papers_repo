from torch.nn import Module

from .encoders import *
from .common import *
from .diffusion import *


class GaussianVAE(Module):
    def __init__(self, args):
        super().__init__()
        self.z = None
        self.log_var = 1
        self.encoder = PointNetEncoder(args.latent_dim)
        self.diffusion = DiffusionPoint()

    def forward(self, x):
        z_mean, self.z_log_var = self.encoder(x)
        self.z = reparameterize_gaussian(z_mean, self.z_log_var)

    def get_loss(self, kl_weight=1.0):
        # get KL by diffusion
        log_pz = standard_normal_logprob(self.z).sum(dim=1)
        entropy = gaussian_entropy(log_var=self.z_log_var)
        loss_prior = (-log_pz - entropy).mean()
        # get KL of second item.
        loss_reverse = self.diffusion.get_loss()
        loss = kl_weight * loss_prior + loss_reverse

        # record entropy/log_pz/loss_reverse
        return loss

    def sample(self):
        pass
