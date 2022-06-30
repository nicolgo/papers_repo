import torch
from torch.optim.lr_scheduler import LambdaLR
import numpy as np


def reparameterize_gaussian(mean, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn(std.size()).to(mean)
    return mean + std * eps


def gaussian_entropy(log_var):
    const = 0.5 * float(log_var.size(1)) * (1. + np.log(np.pi * 2))
    ent = 0.5 * log_var.sum(dim=1, keepdim=False) + const
    return ent


def standard_normal_logprob(z):
    dim = z.size(-1)
    log_z = -0.5 * dim * np.log(2 * np.pi)
    return log_z - z.pow(2) / 2


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
