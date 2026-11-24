"""
HyperDeepONet: DeepONet with a hypernetwork trunk.

Key ideas from Lee & Shin:
- Branch net outputs ARE the trunk net's weights/biases (hypernetwork).
- No learned parameters in the trunk — all trunk params come from the branch output.
"""

import torch
import math
import torch.nn as nn


class LinearHyperDeepONet(nn.Module):
    def __init__(self, branch_dim=674, trunk_dim=2, hidden_dim=46,
                 num_outputs=4, trunk_depth=3, branch_depth=3,
                 activation='GELU'):
        super().__init__()

        if activation == 'Tanh':
            act = nn.Tanh
            self._trunk_act = torch.tanh
        elif activation == 'GELU':
            act = nn.GELU
            self._trunk_act = nn.functional.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Trunk architecture: [trunk_dim, hidden, ..., hidden, num_outputs]
        self.trunk_dims = [trunk_dim] + [hidden_dim] * trunk_depth + [num_outputs]

        # Total parameters needed to construct the trunk net
        t_para = 0
        for i in range(len(self.trunk_dims) - 1):
            t_para += self.trunk_dims[i] * self.trunk_dims[i + 1] + self.trunk_dims[i + 1]

        t_para_w = 0
        for i in range(len(self.trunk_dims) - 1):
            t_para_w += self.trunk_dims[i] * self.trunk_dims[i + 1]

        t_para_b = 0
        for i in range(len(self.trunk_dims) - 1):
            t_para_b += self.trunk_dims[i + 1]


        self.W_linear = nn.Parameter(torch.randn(t_para_w, branch_dim) / math.sqrt(branch_dim))
        self.b_linear = nn.Parameter(torch.randn(t_para_b, branch_dim) / math.sqrt(branch_dim))

        # Branch: single network → t_para (trunk weights/biases)
        branch_dims = [branch_dim] + [hidden_dim] * branch_depth + [t_para]
        self.branch_net = _linear_map(branch_dims, act, self.W_linear, self.b_linear)

        self.num_outputs = num_outputs

        


    def _branch_forward(self, x):
        """Run branch net on x → trunk parameters."""
        return self.branch_net(x)  # [B, t_para]

    def _trunk_forward(self, params_w, params_b, x_trunk):
        """Hypernetwork trunk: params → weights/biases → forward pass."""

        B = params_w.shape[0]
        # Normalize to 3D: [B, N, trunk_dim]
        if x_trunk.dim() == 2:
            x_trunk = x_trunk.unsqueeze(0).expand(B, -1, -1)
        elif x_trunk.shape[0] != B:
            raise ValueError("Branch/trunk batch size mismatch")

        N = x_trunk.shape[1]
        y = x_trunk  # [B, N, trunk_dim]
        start_w = 0
        start_b = 0

        for i in range(len(self.trunk_dims) - 2):
            d_in, d_out = self.trunk_dims[i], self.trunk_dims[i + 1]

            w_sz = d_in * d_out
            weight = params_w[:, start_w:start_w + w_sz].reshape(B, d_out, d_in)
            start_w += w_sz
            bias = params_b[:, start_b:start_b + d_out].reshape(B, 1, d_out)
            start_b += d_out

            y = torch.einsum("bij,bgj->bgi", weight, y) + bias  # [B, N, d_out]
            y = self._trunk_act(y)

        # Last layer: no activation
        d_in, d_out = self.trunk_dims[-2], self.trunk_dims[-1]
        w_sz = d_in * d_out
        weight = params_w[:, start_w:start_w + w_sz].reshape(B, d_out, d_in)
        start_w += w_sz
        bias = params_b[:, start_b:start_b + d_out].reshape(B, 1, d_out)

        y = torch.einsum("bij,bgj->bgi", weight, y) + bias  # [B, N, num_outputs]
        return y

    def forward(self, x_branch, x_trunk):
        """
        Args:
            x_branch: [B, branch_dim]  sensor values
            x_trunk:  [N, trunk_dim] or [B, N, trunk_dim]  query coordinates

        Returns:
            [B, N, num_outputs]
        """
        params_w, params_b = self._branch_forward(x_branch)  # [B, t_para]
        return self._trunk_forward(params_w, params_b, x_trunk)


class _MLP(nn.Module):
    """Simple fully-connected stack: Linear → Act → ... → Linear."""
    def __init__(self, dims, act):
        super().__init__()
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(act())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class _linear_map(nn.Module):
    """Simple fully-connected stack: Linear → Act → ... → Linear."""
    def __init__(self, dims, act, weight, bias):
        super().__init__()

        self.weight = weight
        self.bias = bias

    def forward(self, x):

        Wx = torch.einsum("wa, ba -> bw", self.weight, x)
        b = torch.einsum("wa, ba -> bw", self.bias, x)

        return Wx, b
