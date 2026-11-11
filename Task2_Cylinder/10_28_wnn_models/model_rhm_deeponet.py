"""
RHM-DeepONet: same two-sub-trunk idea as RM-DeepONet (model_rm_deeponet.py)
-- a Ridgelet trunk + an MLP trunk, combined via learnable scalars a, b --
but the MLP sub-trunk is HYPERNETWORK-GENERATED instead of directly
learned, exactly mirroring model_whm_deeponet.py with the WNN sub-trunk
swapped for a Ridgelet sub-trunk.

    trunk(x) = a * RidgeletTrunk(x) + b * HyperMLPTrunk(x; params(u))
"""

import torch
import torch.nn as nn

from model_r_deeponet import RidgeletTrunk


class HyperMLPTrunk(nn.Module):
    """Trunk MLP whose weights/biases are supplied externally (per-sample),
    not stored as nn.Parameters. Identical to model_whm_deeponet.py's."""

    def __init__(self, trunk_dim, hidden_dim, num_basis, depth, activation):
        super().__init__()
        self.trunk_dims = [trunk_dim] + [hidden_dim] * depth + [num_basis]
        self._act = torch.tanh if activation == 'Tanh' else torch.nn.functional.gelu

        self.param_size = 0
        for i in range(len(self.trunk_dims) - 1):
            self.param_size += self.trunk_dims[i] * self.trunk_dims[i + 1] + self.trunk_dims[i + 1]

    def forward(self, params, x_trunk):
        B = params.shape[0]
        y = x_trunk.unsqueeze(0).expand(B, -1, -1)  # [B, N, trunk_dim]
        start = 0
        n_layers = len(self.trunk_dims) - 1
        for i in range(n_layers):
            d_in, d_out = self.trunk_dims[i], self.trunk_dims[i + 1]
            w_sz = d_in * d_out
            weight = params[:, start:start + w_sz].reshape(B, d_out, d_in)
            start += w_sz
            bias = params[:, start:start + d_out].reshape(B, 1, d_out)
            start += d_out
            y = torch.einsum("bij,bgj->bgi", weight, y) + bias
            if i < n_layers - 1:
                y = self._act(y)
        return y  # [B, N, num_basis]


class RHM_DeepONet(nn.Module):
    def __init__(self, branch_dim, trunk_dim, hidden_dim=256, trunk_hidden_dim=None,
                 num_basis=124, num_outputs=4, depth=5, activation='GELU', coord_range=None):
        super().__init__()
        if trunk_hidden_dim is None:
            trunk_hidden_dim = hidden_dim

        if activation == 'GELU':
            self.act = nn.GELU()
        elif activation == 'Tanh':
            self.act = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.hyper_trunk = HyperMLPTrunk(trunk_dim, trunk_hidden_dim, num_basis, depth, activation)
        t_para = self.hyper_trunk.param_size

        backbone = []
        backbone.extend([nn.Linear(branch_dim, hidden_dim), self.act])
        for _ in range(depth - 1):
            backbone.extend([nn.Linear(hidden_dim, hidden_dim), self.act])
        self.branch_backbone = nn.Sequential(*backbone)
        self.head_coeffs = nn.Linear(hidden_dim, num_basis * num_outputs)
        self.head_trunk_params = nn.Linear(hidden_dim, t_para)

        self.trunk_ridgelet = RidgeletTrunk(trunk_dim, num_basis, coord_range=coord_range)

        self.a = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(1.0))

        self.num_outputs = num_outputs
        self.num_basis = num_basis

    def forward(self, x_branch, x_trunk):
        feat = self.branch_backbone(x_branch)         # [B, hidden_dim]
        B_out = self.head_coeffs(feat)                 # [B, num_basis * num_outputs]
        trunk_params = self.head_trunk_params(feat)     # [B, t_para]

        T_ridgelet = self.trunk_ridgelet(x_trunk)          # [N, num_basis]
        T_hyper = self.hyper_trunk(trunk_params, x_trunk)  # [B, N, num_basis]

        T_out = self.a * T_ridgelet.unsqueeze(0) + self.b * T_hyper  # [B, N, num_basis]

        B_out_reshaped = B_out.view(-1, self.num_outputs, self.num_basis)

        prediction = torch.einsum("bkh, bnh -> bkn", B_out_reshaped, T_out)
        prediction = prediction.permute(0, 2, 1)

        return prediction
