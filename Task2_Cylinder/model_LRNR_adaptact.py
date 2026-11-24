"""
Ablation of LRNR: adds Fusion_DeepONet-style adaptive activation parameters
to the trunk's activated layers.

    plain LRNR:    y = act(y)                                        (GELU/Tanh, no learnable params)
    this variant:  y = act(10*a_l*y + c_l) + 10*a1_l*sin(10*F1_l*y + c1_l)

This is exactly Fusion_DeepONet's own adaptive activation form
(model_fusion_deeponet.py: `act(10*a*z+c) + 10*a1*act2(10*F1*z+c1)`,
act2 = sin), with the SAME per-parameter init Fusion_DeepONet uses:
a_l = c_l = F1_l = 0.1, a1_l = c1_l = 0.0 (so at init the sin term
contributes exactly 0 -- matches the original's own asymmetric init,
confirmed correct earlier in this session's Gabor/phi investigation).

One independent (a_l, c_l, F1_l, a1_l, c1_l) tuple per activated trunk
layer: the first_net layer + the (trunk_depth - 1) middle conditional
layers = trunk_depth activated layers total. LRNR's last layer has no
activation to begin with and is left untouched, same as plain LRNR.

U_l, V_l, B_l, branch net, rank s, s_w_offset are all reused unchanged
from LRNR. `linear_branch=True` additionally swaps the branch MLP for a
single nn.Linear (same as model_LRNR_linearbranch.py), so the two
ablations can be combined in one run.
"""

import torch
import torch.nn as nn

from model_LRNR import LRNR


class LRNR_AdaptAct(LRNR):
    def __init__(self, branch_dim=674, trunk_dim=2, hidden_dim=46,
                 num_outputs=4, trunk_depth=3, branch_depth=3,
                 activation='GELU', s=12, s_w_offset=0.0, linear_branch=False):
        super().__init__(branch_dim=branch_dim, trunk_dim=trunk_dim,
                          hidden_dim=hidden_dim, num_outputs=num_outputs,
                          trunk_depth=trunk_depth, branch_depth=branch_depth,
                          activation=activation, s=s, s_w_offset=s_w_offset)

        if linear_branch:
            t_para = self.branch_net.net[-1].out_features
            self.branch_net = nn.Linear(branch_dim, t_para)

        # Activated layers: first_net's layer (1) + middle conditional layers (trunk_depth - 1).
        n_act_layers = trunk_depth
        self.at = nn.Parameter(torch.full((n_act_layers,), 0.1))
        self.ct = nn.Parameter(torch.full((n_act_layers,), 0.1))
        self.F1t = nn.Parameter(torch.full((n_act_layers,), 0.1))
        self.a1t = nn.Parameter(torch.zeros(n_act_layers))
        self.c1t = nn.Parameter(torch.zeros(n_act_layers))
        self._base_act_fn = torch.tanh if activation == 'Tanh' else nn.functional.gelu

    def _adaptive_act(self, y, layer_idx):
        a, c = self.at[layer_idx], self.ct[layer_idx]
        F1, a1, c1 = self.F1t[layer_idx], self.a1t[layer_idx], self.c1t[layer_idx]
        return self._base_act_fn(10 * a * y + c) + 10 * a1 * torch.sin(10 * F1 * y + c1)

    def _trunk_forward(self, params, x_trunk):
        B = params.shape[0]
        if x_trunk.dim() == 2:
            x_trunk = x_trunk.unsqueeze(0).expand(B, -1, -1)
        _, N, _ = x_trunk.shape
        y = x_trunk

        y = self.trunk_net.first_net[0](y)  # Linear only; replace its built-in act() with ours
        y = self._adaptive_act(y, 0)

        s = self.s
        for i in range(1, len(self.trunk_dims) - 2):
            layer_idx = i - 1
            offset = layer_idx * 2 * s
            s_w_l = params[:, offset:offset + s] + self.s_w_offset
            s_b_l = params[:, offset + s:offset + 2 * s]

            weight1 = self.trunk_net.hidden_0_net[2 * layer_idx].weight   # U_l
            weight2 = self.trunk_net.hidden_1_net[2 * layer_idx].weight  # V_l^T
            bias_basis = self.trunk_net.bias_basis_net[layer_idx].weight  # B_l

            w_z = torch.einsum("hs,bs->bhs", weight1, s_w_l)
            weight = torch.einsum("bhs,sr->bhr", w_z, weight2)
            bias = torch.einsum("hs,bs->bh", bias_basis, s_b_l)

            y = torch.einsum("bhr,bnr->bnh", weight, y)
            y = y + bias.unsqueeze(1)
            y = self._adaptive_act(y, layer_idx + 1)  # +1: index 0 was first_net's layer

        y = self.trunk_net.last_net(y)
        return y
