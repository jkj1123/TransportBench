"""
Bilinear_Fusion_DeepONet: Bilinear_DeepONet's low-rank U/V/C bilinear
branch-trunk interaction, combined with Fusion_DeepONet's per-layer scale/
shift activation parameters (ab, cb on the branch side; at, ct on the trunk
side), actually wired into forward() this time -- model_bilinear_deeponet.py
declares the branch-side scalars but never uses them (dead parameters,
confirmed via grad-norm diagnostic: grad_ab_cb_scalars == 0 every epoch).

No sin/act2 term here (removed by request) -- just the scale+shift half of
Fusion_DeepONet's activation:
    x = act(10*a*lin + c)
applied to branch_lin and trunk_lin respectively, in place of the plain
self.act(branch_lin) / self.act(trunk_lin) in Bilinear_DeepONet.
"""
import torch
import torch.nn as nn


class Bilinear_Fusion_DeepONet(nn.Module):
    def __init__(self, branch_dim=1, trunk_dim=2, hidden_dim=280, num_outputs=10, depth=5, activation='GELU'):
        super().__init__()

        if activation == 'GELU':
            self.act = nn.GELU()
        elif activation == 'Tanh':
            self.act = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.L = depth + 1
        self.r = hidden_dim

        # Fusion-deeponet activation parameters -- branch side
        self.ab = nn.Parameter(torch.full((self.L,), 0.1))
        self.cb = nn.Parameter(torch.full((self.L,), 0.1))

        # Fusion-deeponet activation parameters -- trunk side
        self.at = nn.Parameter(torch.full((self.L,), 0.1))
        self.ct = nn.Parameter(torch.full((self.L,), 0.1))

        self.U = nn.Parameter(
            torch.eye(hidden_dim).unsqueeze(0).repeat(self.L, 1, 1)
        )
        self.V = nn.Parameter(
            torch.eye(hidden_dim).unsqueeze(0).repeat(self.L, 1, 1)
        )
        # Fixed (non-learnable), shared across all L layers -- buffer so it
        # moves with .to(device) and is included in state_dict.
        self.register_buffer("C", torch.eye(hidden_dim))

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
            x_branch = self.act(10*self.ab[i]*branch_lin + self.cb[i])
            skip.append(x_branch)

        for i in range(1, self.L - 1):
            skip[i] = skip[i-1] + skip[i]

        reduced_skip = [
            skip[i] @ self.U[i]
            for i in range(self.L - 1)
        ]

        for i in range(self.L - 1):
            trunk_lin = self.trunk_net[i](x_trunk)
            x_trunk = self.act(10*self.at[i]*trunk_lin + self.ct[i])

            x_reduced_trunk = torch.einsum('bik,kr->bir', x_trunk, self.V[i, :, :])
            x_reduced_trunk = torch.einsum('br,bir->bir', reduced_skip[i], x_reduced_trunk)
            x_trunk = torch.einsum('rk,bir->bik', self.C, x_reduced_trunk)

        x_branch = self.branch_net[-1](x_branch)
        x_trunk = self.trunk_net[-1](x_trunk)

        B_out_reshaped = x_branch.view(-1, self.num_outputs, self.hidden_dim)
        prediction = torch.einsum("bnk, bik -> bin", B_out_reshaped, x_trunk)
        return prediction
