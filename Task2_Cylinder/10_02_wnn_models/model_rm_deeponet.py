"""
RM-DeepONet: DeepONet whose trunk is split into TWO parallel sub-trunks that
see the same input coordinate -- a Ridgelet trunk and a plain MLP trunk --
whose outputs are combined via a LEARNABLE weighted sum to form the final
trunk basis:

    trunk(x) = a * RidgeletTrunk(x) + b * MLPTrunk(x)   (both R^trunk_dim -> R^num_basis)

with scalar a, b learned jointly with the rest of the network (init a=b=1).
Direct analogue of model_wm_deeponet.py, with the WNN sub-trunk swapped for
a Ridgelet sub-trunk (see model_r_deeponet.py).
"""

import torch
import torch.nn as nn

from model_r_deeponet import RidgeletTrunk


class MLPTrunk(nn.Module):
    """Same layer pattern as WM-DeepONet's MLPTrunk."""

    def __init__(self, trunk_dim, hidden_dim, num_basis, depth, act):
        super().__init__()
        layers = []
        layers.extend([nn.Linear(trunk_dim, hidden_dim), act])
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), act])
        layers.extend([nn.Linear(hidden_dim, num_basis), act])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class RM_DeepONet(nn.Module):
    def __init__(self, branch_dim, trunk_dim, hidden_dim=256, num_basis=124,
                 num_outputs=4, depth=5, activation='GELU', coord_range=None):
        super().__init__()

        if activation == 'GELU':
            self.act = nn.GELU()
        elif activation == 'Tanh':
            self.act = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        branch_layers = []
        branch_layers.extend([nn.Linear(branch_dim, hidden_dim), self.act])
        for _ in range(depth - 1):
            branch_layers.extend([nn.Linear(hidden_dim, hidden_dim), self.act])
        branch_layers.append(nn.Linear(hidden_dim, num_basis * num_outputs))
        self.branch_net = nn.Sequential(*branch_layers)

        self.trunk_ridgelet = RidgeletTrunk(trunk_dim, num_basis, coord_range=coord_range)
        self.trunk_mlp = MLPTrunk(trunk_dim, hidden_dim, num_basis, depth, self.act)

        self.a = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(1.0))

        self.num_outputs = num_outputs
        self.num_basis = num_basis

    def forward(self, x_branch, x_trunk):
        B_out = self.branch_net(x_branch)  # [Batch, num_basis * num_outputs]

        T_out = self.a * self.trunk_ridgelet(x_trunk) + self.b * self.trunk_mlp(x_trunk)  # [N, num_basis]

        B_out_reshaped = B_out.view(-1, self.num_outputs, self.num_basis)

        prediction = torch.einsum("bkh, nh -> bkn", B_out_reshaped, T_out)
        prediction = prediction.permute(0, 2, 1)

        return prediction
