import torch
import numpy as np
from torch.nn import Module, Linear
from torch.optim.lr_scheduler import LambdaLR


def reparameterize_gaussian(mean, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn(std.size()).to(mean)
    return mean + std * eps


def gaussian_entropy(logvar):
    first_term = 0.5 * float(logvar.size(1)) * (1 + np.log(np.pi * 2))
    total = first_term + 0.5 * logvar.sum(dim=1, keepdim=False)
    return total


def standard_normal_logprob(z):
    dim = z.size(-1)
    log_z = -0.5 * dim * np.log(2 * np.pi)
    return log_z - z.pow(2) / 2


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


def get_linear_scheduler(optimizer, start_epoch, end_epoch, start_lr, end_lr):
    def lr_func(epoch):
        if epoch <= start_epoch:
            return 1.0
        elif epoch <= end_epoch:
            total = end_epoch - start_epoch
            delta = epoch - start_epoch
            frac = delta / total
            return (1 - frac) * 1.0 + frac * (end_lr / start_lr)
        else:
            return end_lr / start_lr

    return LambdaLR(optimizer, lr_lambda=lr_func)
