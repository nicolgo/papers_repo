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
        self.z_mean = None
        self.loss_reverse = None
        self.encoder = PointNetEncoder(args.latent_dim)
        self.flow = build_latent_flow(args)
        self.diffusion = DiffusionPoint(z_dim=args.latent_dim, device=args.device)

    def forward(self, x):
        self.z_mean, self.z_log_var = self.encoder(x)
        self.z = reparameterize_gaussian(self.z_mean, self.z_log_var)
        self.w, self.delta_log_pw = self.flow(self.z, torch.zeros([x.size(0), 1]).to(self.z), reverse=False)
        self.loss_reverse = self.diffusion(x, self.z)
        return self.z, self.loss_reverse

    def get_loss(self, kl_weight=0.001, writer=None, iteration_id=None):
        # TODO: why the expectation did not calculate mean?
        # (B,1) Monte Carlo method: (1/n)*sum(f(X)) = Ep[f(x)]
        log_pw = standard_normal_logprob(self.w).sum(dim=1, keepdim=True)
        log_pz = log_pw - self.delta_log_pw
        # (B,1), H(q(z|x0)): entropy of multivariate normal distribution
        entropy = gaussian_entropy(log_var=self.z_log_var)

        loss_prior = (-log_pz - entropy).mean()
        loss = kl_weight * loss_prior + self.loss_reverse
        # record entropy/log_pz/loss_reverse
        if writer is not None:
            writer.add_scalar('train/loss_entropy', -entropy.mean(), iteration_id)
            writer.add_scalar('train/loss_prior', -log_pz.mean(), iteration_id)
            writer.add_scalar('train/loss_diffusion', self.loss_reverse, iteration_id)
            writer.add_scalar('train/z_mean', self.z_mean.mean(), iteration_id)
            writer.add_scalar('train/z_mag', self.z_mean.abs().max(), iteration_id)
            writer.add_scalar('train/z_var', (0.5 * self.z_log_var).exp().mean(), iteration_id)
        return loss

    def sample(self, w, num_points, truncate_std=None):
        if truncate_std is not None:
            w = truncated_normal_(w, mean=0, std=1, trunc_std=truncate_std)
        z_context = self.flow(w, reverse=True).view(w.size(0), -1)
        samples = self.diffusion.reverse_sample(num_points, z_context)
        return samples
