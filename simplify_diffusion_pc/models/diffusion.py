import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter, ModuleList, Linear
import numpy as np


class ConcatSquashLinear(Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        super().__init__()
        self._layer = Linear(dim_in, dim_out)
        self._hyper_bias = Linear(dim_ctx, dim_out, bias=False)
        self._hyper_gate = Linear(dim_ctx, dim_out)

    def forward(self, input, ctx):
        gate = torch.sigmoid(self._hyper_gate(ctx))  # sigmoid(W2 * c + b2)
        bias = self._hyper_bias(ctx)  # W3 * c
        out = self._layer(input)  # W1 * h(l) + b1
        # h(l+1) = (W1 * h(l) + b1) * sigmoid(W2 * c + b2) + W3 * c
        ret = out * gate + bias
        return ret


class PointWiseNet(Module):
    def __init__(self, z_context_dim, residual):
        super().__init__()
        self.residual = residual
        self.layers = ModuleList([
            ConcatSquashLinear(3, 128, z_context_dim + 3),
            ConcatSquashLinear(128, 256, z_context_dim + 3),
            ConcatSquashLinear(256, 512, z_context_dim + 3),
            ConcatSquashLinear(512, 256, z_context_dim + 3),
            ConcatSquashLinear(256, 128, z_context_dim + 3),
            ConcatSquashLinear(128, 3, z_context_dim + 3),

        ])

    def forward(self, x, beta, z_context):
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)  # (B, 1, 1)
        z_context = z_context.view(batch_size, 1, -1)  # (B, 1, z_dim)

        beta_with_z = torch.cat([beta, torch.sin(beta), torch.cos(beta), z_context], dim=-1)

        out_put = x
        for i, layer in enumerate(self.layers):
            out_put = layer(ctx=beta_with_z, input=out_put)
            if i < (len(self.layers) - 1):
                out_put = F.leaky_relu(out_put)

        if self.residual:
            return x + out_put
        else:
            return x


class DiffusionPoint(Module):
    def __init__(self, z_dim=256, device=torch.device('cpu')):
        super().__init__()
        self.num_steps = 100
        self.betas = torch.linspace(0.0001, 0.02, self.num_steps)
        self.betas = torch.cat([torch.zeros([1]), self.betas], dim=0).to(device)  # add zero to make x(t=0) == x0
        self.alpha = 1. - self.betas
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.alpha_bar_sqrt = torch.sqrt(self.alpha_bar).view(-1, 1, 1)
        self.one_minus_alpha_bar_sqrt = torch.sqrt(1 - self.alpha_bar).view(-1, 1, 1)

        self.gamma = torch.zeros_like(self.betas)
        for t in range(1, self.betas.size(0)):
            self.gamma[t] = ((1 - self.alpha_bar[t - 1]) / (1 - self.alpha_bar[t])) * self.betas[t]
        self.gamma = torch.sqrt(self.gamma)

        # define network
        self.net = PointWiseNet(z_context_dim=z_dim, residual=True)

    def sample_t_randomly(self, batch_size):
        # ts = np.random.choice(np.arange(1, self.num_steps + 1), batch_size)
        # return ts.tolist()
        return torch.randint(1, self.num_steps + 1, (batch_size,))

    def forward(self, x_0, z_context):
        batch_size, _, point_dim = x_0.size()
        noise = torch.randn_like(x_0)
        # calculate xt|(x0,ε)
        t_batches = self.sample_t_randomly(batch_size=batch_size)
        mean = self.alpha_bar_sqrt[t_batches].view(-1, 1, 1)
        delta = self.one_minus_alpha_bar_sqrt[t_batches].view(-1, 1, 1)

        x_t = mean * x_0 + delta * noise
        beta = self.betas[t_batches]

        noise_theta = self.net(x_t, beta=beta, z_context=z_context)

        loss = F.mse_loss(noise.view(-1, 3), noise_theta.view(-1, 3), reduction='mean')
        return loss

    def diffusion_process(self, x_0: torch.Tensor):
        noise = torch.randn_like(x_0)
        x_0_t = self.alpha_bar_sqrt * x_0 + self.one_minus_alpha_bar_sqrt * noise
        return x_0_t

    def reverse_sample(self, num_points, z_context, ret_traj=False):
        batch_size = z_context.size(0)
        x_t = torch.randn([z_context.size(0), num_points, 3]).to(z_context.device)
        traj = {self.num_steps: x_t}
        for t in range(self.num_steps, 0, -1):
            x_t = traj[t]
            beta = self.betas[[t] * batch_size]
            e_theta = self.net(x_t, beta=beta, z_context=z_context)

            c0 = 1. / torch.sqrt(self.alpha[t])
            c1 = self.betas[t] / torch.sqrt(1 - self.alpha_bar[t])
            epsilon = torch.randn_like(x_t)
            x_before = c0 * (x_t - c1 * e_theta) + self.gamma[t] * epsilon
            traj[t - 1] = x_before.detach()
            traj[t] = traj[t].cpu()
            if not ret_traj:
                del traj[t]

        if ret_traj:
            return traj
        else:
            return traj[0]

    def reverse_with_x0_xt(self, x0: torch.Tensor):
        pass
