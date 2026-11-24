"""
CoordinateMLP: the simplest possible baseline for comparison against
HyperDeepONet / Fusion-DeepONet / LRNR etc.

    u_hat(x; mu) = f_theta([mu, x])

The condition vector (branch input) and spatial coordinates (trunk input)
are concatenated and fed through one plain MLP. No hypernetwork, no
DeepONet branch-trunk inner product, no low-rank factorization, no FiLM,
no residual connections -- just Linear -> activation -> ... -> Linear on
the concatenated input.

Matches HyperDeepONet's constructor signature and forward() interface
(model_hyperdeeponet.py) so it drops into the same training/eval code.
No input normalization or output denormalization is added here -- x_branch
and x_trunk are consumed exactly as the existing data pipeline provides
them (input_params is already [0,1]-normalized upstream in the .pt file;
grid_coords is left in its raw physical range, e.g. [-3.0, 5.5] for
Task2_Cylinder -- see data_loader_fast.py / cylinder_full_2400_400_seed42.pt).
"""

import torch
import torch.nn as nn


class CoordinateMLP(nn.Module):
    def __init__(self, branch_dim=674, trunk_dim=2, hidden_dim=46,
                 num_outputs=4, trunk_depth=3, branch_depth=3,
                 activation='GELU'):
        super().__init__()
        # branch_depth is accepted only for interface compatibility with
        # HyperDeepONet/LRNR; there is no separate branch network here, so
        # it is unused.
        del branch_depth

        if activation == 'GELU':
            act = nn.GELU
        elif activation == 'Tanh':
            act = nn.Tanh
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.branch_dim = branch_dim
        self.trunk_dim = trunk_dim
        self.num_outputs = num_outputs

        in_dim = branch_dim + trunk_dim
        dims = [in_dim] + [hidden_dim] * trunk_depth + [num_outputs]

        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(act())
        layers.append(nn.Linear(dims[-2], dims[-1]))  # no activation on the output layer
        self.net = nn.Sequential(*layers)

    def forward(self, x_branch, x_trunk):
        """
        Args:
            x_branch: [B, branch_dim]  condition vector (e.g. Kn, Ma)
            x_trunk:  [N, trunk_dim] (shared across the batch) or
                      [B, N, trunk_dim] (per-sample coordinates)

        Returns:
            [B, N, num_outputs]
        """
        B = x_branch.shape[0]
        if x_trunk.dim() == 2:
            N = x_trunk.shape[0]
            x_trunk = x_trunk.unsqueeze(0).expand(B, N, self.trunk_dim)
        else:
            N = x_trunk.shape[1]

        x_branch_exp = x_branch.unsqueeze(1).expand(B, N, self.branch_dim)
        x = torch.cat([x_branch_exp, x_trunk], dim=-1)  # [B, N, branch_dim + trunk_dim]
        return self.net(x)
