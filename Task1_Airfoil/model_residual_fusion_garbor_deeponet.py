"""
Residual_Fusion_Garbor_DeepONet: same real-valued Gabor-activation
Fusion_DeepONet as Task2_Cylinder/model_fusion_garbor_deeponet.py, ported to
Task1_Airfoil dimensions, with exactly ONE structural change from that file:
the trunk-skip gate uses the Residual_Fusion_DeepONet-style residual gate
    x_trunk = x_trunk * (1.0 + skip[i])
instead of the plain product
    x_trunk = x_trunk * skip[i]
used by model_fusion_garbor_deeponet.py. Everything else (GaborActivation
definition, branch/trunk Linear stacks, skip accumulation, final einsum,
independent per-layer branch/trunk Gabor parameters) is unchanged.
"""
import torch
import torch.nn as nn


class GaborActivation(nn.Module):
    """Real-valued Gabor activation: phi(z) = exp(-s^2 * z^2) * cos(w * z).

    w (omega) and s are learnable, layerwise SCALAR parameters (shared by
    every neuron in the layer), initialized to w0=1.0 / s0=1.0. No complex
    arithmetic is involved -- input and output are both real.
    """
    def __init__(self, w0=1.0, s0=1.0):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(float(w0)))
        self.s = nn.Parameter(torch.tensor(float(s0)))

    def forward(self, z):
        return torch.exp(-(self.s ** 2) * z ** 2) * torch.cos(self.w * z)


class Residual_Fusion_Garbor_DeepONet(nn.Module):
    def __init__(self, branch_dim=674, trunk_dim=2, hidden_dim=253, num_outputs=4, depth=5, activation='GELU'):
        """
        Fusion DeepONet for Airfoil Task, with Gabor-activation hidden layers
        and the residual (1 + skip) trunk-skip gate.

        Args:
            branch_dim: 674 (337 geometry points x 2)
            trunk_dim: 2 (x, y)
            hidden_dim: width of hidden layers
            num_outputs: 4 (Physics quantities)
            activation: unused -- every hidden layer uses the real Gabor
                activation (see GaborActivation above); kept as a constructor
                arg only so existing call sites don't need to change.
        """
        super().__init__()

        # One independent (w_l, s_l) Gabor activation per hidden layer, for
        # BranchNet and TrunkNet separately (depth each).
        self.branch_acts = nn.ModuleList([GaborActivation(w0=1.0, s0=1.0) for _ in range(depth)])
        self.trunk_acts = nn.ModuleList([GaborActivation(w0=1.0, s0=1.0) for _ in range(depth)])

        # Fusion-deeponet bookkeeping (unchanged): self.L-1 == depth hidden
        # layers with an activation, plus one final (un-activated) output layer.
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

        for i in range(self.L - 1):
            branch_lin = self.branch_net[i](x_branch)
            x_branch = self.branch_acts[i](branch_lin)
            skip.append(x_branch)

        for i in range(1, self.L - 1):
            skip[i] = skip[i - 1] + skip[i]

        for i in range(self.L - 1):
            trunk_lin = self.trunk_net[i](x_trunk)
            x_trunk = self.trunk_acts[i](trunk_lin)

            # Residual gate: identity path preserves trunk magnitude/gradient
            # (the one intentional difference from model_fusion_garbor_deeponet.py).
            x_trunk = torch.einsum('bk,bik->bik', 1.0 + skip[i], x_trunk)

        x_branch = self.branch_net[-1](x_branch)
        x_trunk = self.trunk_net[-1](x_trunk)

        B_out_reshaped = x_branch.view(-1, self.num_outputs, self.hidden_dim)
        # [Batch, num_outputs, hidden_dim]

        prediction = torch.einsum("bnk, bik -> bin", B_out_reshaped, x_trunk)
        # [Batch, N_points, num_outputs]

        return prediction
