"""
Ablation of model_fusion_garbor_deeponet.py: identical real-valued Gabor
activation phi(z) = exp(-s^2 * z^2) * cos(w * z), but w (omega) and s are
FIXED constants (w0=1.0, s0=1.0, registered as buffers, no gradient) instead
of learnable nn.Parameters. Everything else (branch/trunk Linear stacks,
skip accumulation, final einsum) is byte-for-byte identical to the learnable
version -- this isolates how much of the learnable version's behavior comes
from actually adapting (w, s) per layer vs. just using the Gabor functional
form itself.
"""
import torch
import torch.nn as nn


class FixedGaborActivation(nn.Module):
    """Real-valued Gabor activation with FIXED (non-learnable) w, s."""
    def __init__(self, w0=1.0, s0=1.0):
        super().__init__()
        self.register_buffer('w', torch.tensor(float(w0)))
        self.register_buffer('s', torch.tensor(float(s0)))

    def forward(self, z):
        return torch.exp(-(self.s ** 2) * z ** 2) * torch.cos(self.w * z)


class Fusion_DeepONet(nn.Module):
    def __init__(self, branch_dim=1, trunk_dim=2, hidden_dim=280, num_outputs=10, depth=5, activation='GELU'):
        """
        Same as model_fusion_garbor_deeponet.Fusion_DeepONet, but w/s in every
        Gabor activation are fixed (w0=1.0, s0=1.0), not learned.
        """
        super().__init__()

        self.branch_acts = nn.ModuleList([FixedGaborActivation(w0=1.0, s0=1.0) for _ in range(depth)])
        self.trunk_acts = nn.ModuleList([FixedGaborActivation(w0=1.0, s0=1.0) for _ in range(depth)])

        self.L = depth + 1

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
        if x_trunk.dim() == 2:
            x_trunk = x_trunk.unsqueeze(0).repeat(x_branch.shape[0], 1, 1)

        skip = []

        for i in range(self.L - 1):
            branch_lin = self.branch_net[i](x_branch)
            x_branch = self.branch_acts[i](branch_lin)
            skip.append(x_branch)

        for i in range(1, self.L - 1):
            skip[i] = skip[i - 1] + skip[i]

        for i in range(self.L - 1):
            trunk_lin = self.trunk_net[i](x_trunk)
            x_trunk = self.trunk_acts[i](trunk_lin)

            x_trunk = torch.einsum('bk,bik->bik', skip[i], x_trunk)

        x_branch = self.branch_net[-1](x_branch)
        x_trunk = self.trunk_net[-1](x_trunk)

        B_out_reshaped = x_branch.view(-1, self.num_outputs, self.hidden_dim)

        prediction = torch.einsum("bnk, bik -> bin", B_out_reshaped, x_trunk)

        return prediction
