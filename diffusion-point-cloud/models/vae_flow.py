from torch.nn import Module
from .encoders import *
from .diffusion import *
from .flow import *


class FlowVAE(Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = PointNetEncoder(args.latent_dim)
        self.flow = build_latent_flow(args)
        self.diffusion = DiffusionPoint(
            net=PointwiseNet(point_dim=3, context_dim=args.latent_dim, residual=args.residual),
            var_sched=VarianceSchedule(
                num_steps=args.num_steps,
                beta_1=args.beta_1,
                beta_T=args.beta_T,
                mode=args.sched_mode
            )
        )

    def get_loss(self, x, kl_weight, writer=None, it=None):
        batch_size, _, _ = x.size()
        z_mu, z_sigma = self.encoder(x)

        z = reparameterize_gaussian(mean=z_mu, logvar=z_sigma)



