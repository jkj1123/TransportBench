"""
Ablation of model_fusion_deeponet.py (not modified by this file): removes
the GELU/Tanh term from the activation entirely, keeping only the
learnable-amplitude/frequency/phase sin term:

    original: y = act(10*a*z + c) + 10*a1*sin(10*F1*z + c1)
    this file: y = 10*a1*sin(10*F1*z + c1)

a1 is still zero-initialized (as in the original), so this model also
starts as the identically-zero function and has to bootstrap entirely
through the sin term's own gradient -- same init convention as the
original, just with no GELU term to also contribute from the start.

a/c (the now-unused GELU-term scale/shift) are removed; a1/F1/c1 (sin
amplitude/frequency/phase) are kept exactly as before. Branch/trunk Linear
stacks, skip accumulation, and the final DeepONet inner product are
unchanged. `activation` constructor arg is accepted but unused (kept only
so existing call sites don't need to change).
"""
import torch
import torch.nn as nn


class sin_act(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class Fusion_DeepONet(nn.Module):
    def __init__(self, branch_dim=1, trunk_dim=2, hidden_dim=280, num_outputs=10, depth=5, activation='GELU'):
        super().__init__()

        self.act2 = sin_act()

        # Fusion-deeponet additional parameters (sin term only -- no a/c)
        self.L = depth + 1

        self.a1b = nn.Parameter(torch.zeros(self.L,))
        self.F1b = nn.Parameter(torch.full((self.L,), 0.1))
        self.c1b = nn.Parameter(torch.zeros(self.L,))

        self.a1t = nn.Parameter(torch.zeros(self.L,))
        self.F1t = nn.Parameter(torch.full((self.L,), 0.1))
        self.c1t = nn.Parameter(torch.zeros(self.L,))

        # Branch Net
        self.branch_net = nn.ModuleList()
        self.branch_net.append(nn.Linear(branch_dim, hidden_dim))
        for _ in range(depth - 1):
            self.branch_net.append(nn.Linear(hidden_dim, hidden_dim))
        self.branch_net.append(nn.Linear(hidden_dim, hidden_dim * num_outputs))

        # Trunk Net: [Linear1, Linear2, ..., Last Linear]
        self.trunk_net = nn.ModuleList()
        self.trunk_net.append(nn.Linear(trunk_dim, hidden_dim))
        for _ in range(depth - 1):
            self.trunk_net.append(nn.Linear(hidden_dim, hidden_dim))
        self.trunk_net.append(nn.Linear(hidden_dim, hidden_dim))

        self.num_outputs = num_outputs
        self.hidden_dim = hidden_dim

    def forward(self, x_branch, x_trunk):

        """Forward pass.

        Args:
            x_branch: [Batch, branch_dim]
            x_trunk:  [N_points, trunk_dim]

        Returns:
            [Batch, N_points, num_outputs]
        """

        if x_trunk.dim() == 2:
            # [N, trunk_dim] -> [1, N, trunk_dim] -> [B, N, trunk_dim]
            x_trunk = x_trunk.unsqueeze(0).repeat(x_branch.shape[0], 1, 1)

        skip = []

        for i in range(self.L-1):

            branch_lin = self.branch_net[i](x_branch)
            x_branch = 10*self.a1b[i]*self.act2(10*self.F1b[i]*branch_lin+self.c1b[i])
            skip.append(x_branch)

        for i in range(1,self.L-1):
            skip[i] = skip[i-1]+skip[i]

        for i in range(self.L-1):

            trunk_lin = self.trunk_net[i](x_trunk)
            x_trunk = 10*self.a1t[i]*self.act2(10*self.F1t[i]*trunk_lin+self.c1t[i])

            x_trunk = torch.einsum('bk,bik->bik', skip[i], x_trunk)

        x_branch = self.branch_net[-1](x_branch)
        x_trunk = self.trunk_net[-1](x_trunk)

        B_out_reshaped = x_branch.view(-1, self.num_outputs, self.hidden_dim)
        # [Batch, num_outputs, hidden_dim]

        prediction = torch.einsum("bnk, bik -> bin", B_out_reshaped, x_trunk)
        # [Batch, N_points, num_outputs]

        return prediction
