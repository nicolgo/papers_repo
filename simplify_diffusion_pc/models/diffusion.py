import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter, ModuleList, Linear


class VarianceSchedule(Module):
    def __init__(self):
        super().__init__()
        self.num_steps = 100
        self.beta = torch.linspace(0.0001, 0.02, self.num_steps)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)


class ConcatSquashLinear(Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        super().__init__()
        self._layer = Linear(dim_in, dim_out)
        self._hyper_bias = Linear(dim_ctx, dim_out, bias=False)
        self._hyper_gate = Linear(dim_ctx, dim_out)

    def forward(self, ctx, x):
        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        out = self._layer(x)
        ret = out * gate + bias
        return ret


class PointWiseNet(Module):
    def __int__(self):
        pass


class DiffusionPoint(Module):
    def __init__(self):
        super().__init__()
        self.num_steps = 100
        self.beta = torch.linspace(0.0001, 0.02, self.num_steps)
        # add zero to make x(t=0) == x0
        self.beta = torch.cat([torch.zeros([1]), self.beta], dim=0)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def forward(self):
        pass

    def get_loss(self):
        pass

    def diffusion_process(self, x_0: torch.Tensor):
        noise = torch.randn_like(x_0)
        alpha_bar_sqrt = torch.sqrt(self.alpha_bar).view(-1, 1, 1)
        one_minus_alpha_bar_sqrt = torch.sqrt(1 - self.alpha_bar).view(-1, 1, 1)
        x_0_t = alpha_bar_sqrt * x_0 + one_minus_alpha_bar_sqrt * noise
        return x_0_t

    def reverse_sample(self):
        pass
