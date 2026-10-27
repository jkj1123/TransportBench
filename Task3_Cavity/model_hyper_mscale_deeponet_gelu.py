"""
HyperMscaleDeepONetGELU: same as HyperMscaleDeepONet, but the trunk's
internal B-spline activation (`_phi`) is swapped for plain GELU.

This isolates the trunk-activation cost identified while profiling
HyperDeepONet vs HyperMscaleDeepONet: both hypernetwork trunks generate
per-sample weights and apply them via the same einsum pattern, but
HyperMscaleDeepONet's `_phi` (relu(x)**2 - 3*relu(x-1)**2 + ...) has a
much more expensive backward pass than HyperDeepONet's plain GELU.
Everything else (branch net, weight-generation, einsum trunk forward,
log-scale) is identical to model_hyper_mscale_deeponet.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _compute_weight_bias(dims):
    """Total parameter count for a linear stack of given dims (weights + biases)."""
    total = 0
    for i in range(len(dims) - 1):
        total += dims[i] * dims[i + 1] + dims[i + 1]
    return total


class _MLP(nn.Module):
    """Fully-connected stack: Linear -> Act -> ... -> Linear."""
    def __init__(self, dims, act):
        super().__init__()
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(act)
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class HyperMscaleDeepONetGELU(nn.Module):
    """HyperMscaleDeepONet with the trunk's B-spline activation replaced by GELU.

    Args:
        branch_dim:   Input dimension of the branch net (sensor values).
        trunk_dim:    Input dimension of the trunk net (coordinates).
        hidden_dim:   Width of hidden layers.
        num_outputs:  Number of output channels.
        depth:        Number of hidden layers in trunk FNN and branch net.
        activation:   'GELU' or 'Tanh' (branch activation; trunk always uses GELU here).
    """
    def __init__(self, branch_dim, trunk_dim, hidden_dim=256, num_outputs=4,
                 depth=4, activation='GELU'):
        super().__init__()

        if activation == 'GELU':
            act = nn.GELU()
        elif activation == 'Tanh':
            act = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.num_outputs = num_outputs

        trunk_dims = [trunk_dim] + [hidden_dim] * depth
        trunk_params = _compute_weight_bias(trunk_dims)

        output_dims = [hidden_dim, num_outputs]
        output_params = _compute_weight_bias(output_dims)

        t_para = trunk_params + output_params + 1  # +1 for log_scale

        self.branch_net = _MLP([branch_dim] + [hidden_dim] * depth + [t_para], act)

        self._trunk_dims = trunk_dims
        self._output_dims = output_dims

    @staticmethod
    def _apply_layer(params, x, d_in, d_out, start, act_fn=None):
        B = params.shape[0]
        w_sz = d_in * d_out
        weight = params[:, start:start + w_sz].reshape(B, d_out, d_in)
        start += w_sz
        bias = params[:, start:start + d_out].reshape(B, 1, d_out)
        start += d_out
        y = torch.einsum("bij,bgj->bgi", weight, x) + bias
        if act_fn is not None:
            y = act_fn(y)
        return y, start

    def _trunk_forward(self, params, x_trunk):
        if x_trunk.dim() == 2:
            x_trunk = x_trunk.unsqueeze(0)
        B = params.shape[0]

        log_scale = params[:, -1:]
        scale = torch.exp(log_scale)
        y = scale.view(B, 1, 1) * x_trunk

        start = 0
        for i in range(len(self._trunk_dims) - 1):
            d_in = self._trunk_dims[i]
            d_out = self._trunk_dims[i + 1]
            y, start = self._apply_layer(params, y, d_in, d_out, start,
                                         act_fn=F.gelu)  # <-- only change vs. model_hyper_mscale_deeponet.py

        d_oin, d_oout = self._output_dims[0], self._output_dims[1]
        y, start = self._apply_layer(params, y, d_oin, d_oout, start,
                                     act_fn=None)
        return y

    def forward(self, x_branch, x_trunk):
        params = self.branch_net(x_branch)
        return self._trunk_forward(params, x_trunk)
