"""
Ablation of Fusion_DeepONet: removes ALL ten per-layer scalar "minor"
parameters (branch: ab, cb, a1b, F1b, c1b; trunk: at, ct, a1t, F1t, c1t) from
both the branch and trunk nets, replacing the custom
act(10*a*x+c) + 10*a1*sin(10*F1*x+c1) activation with a plain self.act(x).
Skip connections (skip[i] = skip[i-1]+skip[i], and the branch/trunk gating
einsum) are left untouched -- this isolates the scalar parameters specifically,
as opposed to the skip structure (see model_fusion_deeponet's 'noskip' variant).

Identical to model_fusion_deeponet.Fusion_DeepONet otherwise (same branch/trunk
net shapes, same output projection), so param count differs only by the
removed 10*L scalars (negligible at the ~100K tier).
"""
import torch
import torch.nn as nn


class Fusion_DeepONet_NoScalars(nn.Module):
    def __init__(self, branch_dim=1, trunk_dim=2, hidden_dim=280, num_outputs=10, depth=5, activation='GELU'):
        super().__init__()

        if activation == 'GELU':
            self.act = nn.GELU()
        elif activation == 'Tanh':
            self.act = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.L = depth + 1

        # Branch Net
        self.branch_net = nn.ModuleList()
        self.branch_net.append(nn.Linear(branch_dim, hidden_dim))
        for _ in range(depth - 1):
            self.branch_net.append(nn.Linear(hidden_dim, hidden_dim))
        self.branch_net.append(nn.Linear(hidden_dim, hidden_dim * num_outputs))

        # Trunk Net
        self.trunk_net = nn.ModuleList()
        self.trunk_net.append(nn.Linear(trunk_dim, hidden_dim))
        for _ in range(depth - 1):
            self.trunk_net.append(nn.Linear(hidden_dim, hidden_dim))
        self.trunk_net.append(nn.Linear(hidden_dim, hidden_dim))

        self.num_outputs = num_outputs
        self.hidden_dim = hidden_dim

    def forward(self, x_branch, x_trunk):
        if x_trunk.dim() == 2:
            x_trunk = x_trunk.unsqueeze(0).repeat(x_branch.shape[0], 1, 1)

        skip = []

        for i in range(self.L - 1):
            branch_lin = self.branch_net[i](x_branch)
            x_branch = self.act(branch_lin)
            skip.append(x_branch)

        for i in range(1, self.L - 1):
            skip[i] = skip[i-1] + skip[i]

        for i in range(self.L - 1):
            trunk_lin = self.trunk_net[i](x_trunk)
            x_trunk = self.act(trunk_lin)
            x_trunk = torch.einsum('bk,bik->bik', skip[i], x_trunk)

        x_branch = self.branch_net[-1](x_branch)
        x_trunk = self.trunk_net[-1](x_trunk)

        B_out_reshaped = x_branch.view(-1, self.num_outputs, self.hidden_dim)
        prediction = torch.einsum("bnk, bik -> bin", B_out_reshaped, x_trunk)
        return prediction
