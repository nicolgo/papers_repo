import torch
from torch.nn import Module, Linear


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
