from torch.nn import Module
from .encoders import *


class GaussianVAE(Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = PointNetEncoder(args.latent_dim)
        self.diffusion
