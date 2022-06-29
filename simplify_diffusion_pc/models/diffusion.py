import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter, ModuleList, Linear


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
        pass

    def forward(self):
        pass

    def get_loss(self):
        pass


if __name__ == "__main__":
    layer = ConcatSquashLinear(3, 128, 128 + 3)

    x = torch.randn(2, 2048, 3)
    z = torch.rand(2, 131)
    y = layer(z, x)
    pass
