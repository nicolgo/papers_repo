import torch
import numpy as np
from torch.nn import Module, Linear


def reparameterize_gaussian(mean, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn(std.size()).to(mean)
    return mean + std * eps


def gaussian_entropy(logvar):
    first_term = 0.5 * float(logvar.size(1)) * (1 + np.log(np.pi * 2))
    total = first_term + 0.5 * logvar.sum(dim=1, keepdim=False)
    return total


class ConcatSquashLinear(Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        super(ConcatSquashLinear, self).__init__()
        self._layer = Linear(dim_in, dim_out)
        self._hyper_bias = Linear(dim_ctx, dim_out, bias=False)
        self._hyper_gate = Linear(dim_ctx, dim_out)

    def forward(self, ctx, x):
        # ctx = [t, sin(t), cos(t), z], gate = sigmoid(W2*ctx+b2)
        gate = torch.sigmoid(self._hyper_gate(ctx))
        # bias = W3*ctx
        bias = self._hyper_bias(ctx)
        # ret = (W1x+b1)*(sigmoid(W2*ctx+b2))+W3*ctx
        ret = self._layer(x) * gate + bias
        return ret
