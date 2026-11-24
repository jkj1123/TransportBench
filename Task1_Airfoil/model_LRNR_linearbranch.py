"""
Self-contained port of Task2_Cylinder/low_rank_models/model_LRNR.py +
Task2_Cylinder/model_LRNR_linearbranch.py for Task1_Airfoil. Does not modify
the existing model_LRNR.py (which train_residual_lrnr.py already depends
on) -- this is a pure addition.

LRNR here is identical in behavior to Task1's existing model_LRNR.py when
called without `s`/`s_w_offset` (defaults s=12, s_w_offset=0.0 reproduce the
original exactly); it just also exposes `s` and `s_w_offset` as constructor
args, matching the newer Task2_Cylinder version.

LRNR_LinearBranch replaces the branch net -- a multi-layer MLP
(Linear -> Act -> ... -> Linear) -- with a single plain
nn.Linear(branch_dim, t_para), no activation, no hidden layers:

    plain LRNR:        s_w_l, s_b_l = MLP(x_branch)      (GELU/Tanh hidden layers)
    this variant:      s_w_l, s_b_l = Linear(x_branch)   (no nonlinearity anywhere in the branch)

Everything else (U_l, V_l, B_l, rank s, s_w_offset, trunk activation, trunk
depth, first/last trunk layers, forward()) is unchanged. Motivation: test
whether this much simpler, FiLM/diagonal-gating-style branch (validated as
an unstick fix on Task2_Cylinder) avoids the branch-conditioning collapse
documented in TASK1_RESIDUAL_LEARNING_FAILURE_NOTES.md for Task1's residual
target, where the full MLP branch always collapses once sample count > ~20.
"""

import torch
import torch.nn as nn


class LRNR(nn.Module):
    def __init__(self, branch_dim=674, trunk_dim=2, hidden_dim=46,
                 num_outputs=4, trunk_depth=3, branch_depth=3,
                 activation='GELU', s=12, s_w_offset=0.0):
        super().__init__()
        self.s_w_offset = s_w_offset

        if activation == 'Tanh':
            act = nn.Tanh
            self._trunk_act = torch.tanh
        elif activation == 'GELU':
            act = nn.GELU
            self._trunk_act = nn.functional.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.trunk_dims = [trunk_dim] + [hidden_dim] * trunk_depth + [num_outputs]

        self.s = s
        self.num_hidden_layers = trunk_depth - 1
        t_para = self.num_hidden_layers * 2 * self.s

        branch_dims = [branch_dim] + [hidden_dim] * branch_depth + [t_para]
        self.branch_net = _MLP(branch_dims, act)
        self.trunk_net = _MLP_reduced(self.trunk_dims, self.s, act)

        self.num_outputs = num_outputs

    def _branch_forward(self, x):
        return self.branch_net(x)  # [B, t_para]

    def _trunk_forward(self, params, x_trunk):
        B = params.shape[0]
        if x_trunk.dim() == 2:
            x_trunk = x_trunk.unsqueeze(0).expand(B, -1, -1)

        _, N, _ = x_trunk.shape
        y = x_trunk

        y = self.trunk_net.first_net(y)

        s = self.s
        for i in range(1, len(self.trunk_dims) - 2):
            layer_idx = i - 1
            offset = layer_idx * 2 * s
            s_w_l = params[:, offset:offset + s] + self.s_w_offset
            s_b_l = params[:, offset + s:offset + 2 * s]

            weight1 = self.trunk_net.hidden_0_net[2 * layer_idx].weight
            weight2 = self.trunk_net.hidden_1_net[2 * layer_idx].weight
            bias_basis = self.trunk_net.bias_basis_net[layer_idx].weight

            w_z = torch.einsum("hs,bs->bhs", weight1, s_w_l)
            weight = torch.einsum("bhs,sr->bhr", w_z, weight2)
            bias = torch.einsum("hs,bs->bh", bias_basis, s_b_l)

            y = torch.einsum("bhr,bnr->bnh", weight, y)
            y = y + bias.unsqueeze(1)
            y = self._trunk_act(y)

        y = self.trunk_net.last_net(y)
        return y

    def forward(self, x_branch, x_trunk):
        params = self._branch_forward(x_branch)
        return self._trunk_forward(params, x_trunk)


class LRNR_LinearBranch(LRNR):
    def __init__(self, branch_dim=674, trunk_dim=2, hidden_dim=46,
                 num_outputs=4, trunk_depth=3, branch_depth=3,
                 activation='GELU', s=12, s_w_offset=0.0):
        super().__init__(branch_dim=branch_dim, trunk_dim=trunk_dim,
                          hidden_dim=hidden_dim, num_outputs=num_outputs,
                          trunk_depth=trunk_depth, branch_depth=branch_depth,
                          activation=activation, s=s, s_w_offset=s_w_offset)
        t_para = self.branch_net.net[-1].out_features
        self.branch_net = nn.Linear(branch_dim, t_para)


class _MLP(nn.Module):
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


class _MLP_reduced(nn.Module):
    def __init__(self, dims, diag, act):
        super().__init__()
        first_layers = []
        hidden_layers_0 = []
        hidden_layers_1 = []
        bias_basis_layers = []
        last_layers = []

        first_layers.append(nn.Linear(dims[0], dims[1]))
        first_layers.append(act())

        for i in range(1, len(dims) - 2):
            hidden_layers_0.append(nn.Linear(diag, dims[i + 1], bias=False))
            hidden_layers_0.append(act())
            hidden_layers_1.append(nn.Linear(dims[i], diag, bias=False))
            hidden_layers_1.append(act())
            bias_basis_layers.append(nn.Linear(diag, dims[i + 1], bias=False))
        last_layers.append(nn.Linear(dims[-2], dims[-1]))
        self.first_net = nn.Sequential(*first_layers)
        self.hidden_0_net = nn.Sequential(*hidden_layers_0)
        self.hidden_1_net = nn.Sequential(*hidden_layers_1)
        self.bias_basis_net = nn.ModuleList(bias_basis_layers)
        self.last_net = nn.Sequential(*last_layers)
