from torch.nn import Module

from .diffusion import *


class PureDiffusion(Module):
    def __init__(self, args):
        super().__init__()
        self.loss_diffusion = None
        self.diffusion = DiffusionPoint(z_dim=0, device=args.device)

    def forward(self, x):
        self.loss_diffusion = self.diffusion(x, None)
        return self.loss_diffusion

    def get_loss(self, kl_weight=0.001, writer=None, iteration_id=None):
        """
        only one parameter is useful, others only stay same interface with before.
        """
        loss = self.loss_diffusion
        # record loss_diffusion
        return loss

    def sample(self, z_context, num_points, truncate_std=None):
        samples = self.diffusion.reverse_sample(num_points, z_context=None, batch_size=z_context.size(0),
                                                device=z_context.device)
        return samples
