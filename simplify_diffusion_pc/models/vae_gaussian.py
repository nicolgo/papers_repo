from torch.nn import Module

from .encoders import *
from .common import *


class GaussianVAE(Module):
    def __init__(self, args):
        super().__init__()
        self.z = None
        self.log_var = 1
        self.encoder = PointNetEncoder(args.latent_dim)

    def forward(self, x):
        mean, log_var = self.encoder(x)
        self.z = reparameterize_gaussian(mean, log_var)


    def get_loss(self):
        # get KL by diffusion

        # get KL of second item.

        pass

    def sample(self):
        pass
