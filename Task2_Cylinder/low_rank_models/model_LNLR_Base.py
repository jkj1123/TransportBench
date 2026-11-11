"""
LNLR_Base: ablation variant of LNLR with an added learnable full-rank base
weight at each conditional (hypernetwork-generated) hidden layer.

    LNLR:       W_l(mu) = U_l diag(s_w_l(mu)) V_l^T
    LNLR_Base:  W_l(mu) = W_{0,l} + U_l diag(s_w_l(mu)) V_l^T

W_{0,l} is a plain nn.Parameter, zero-initialized and shared across the
batch (not conditioned on mu, unlike the U_l diag(s_w_l) V_l^T term). Zero
init means the very first forward pass is numerically identical to plain
LNLR; W_{0,l} only starts contributing once gradients move it away from
zero. This isolates whether LNLR's accuracy gap is caused by the
hidden-layer weight being constrained to rank <= s, by letting the model
"escape" that constraint with a full-rank term if it needs to.

Branch net, rank s, activation, trunk depth, first/last trunk layers, and
the U_l/V_l/B_l + s_w_l/s_b_l machinery are reused unchanged from
model_LNLR.py (not modified by this file).
"""

import torch
import torch.nn as nn

from model_LNLR import LNLR


class LNLR_Base(LNLR):
    def __init__(self, branch_dim=674, trunk_dim=2, hidden_dim=46,
                 num_outputs=4, trunk_depth=3, branch_depth=3,
                 activation='GELU'):
        super().__init__(branch_dim=branch_dim, trunk_dim=trunk_dim,
                          hidden_dim=hidden_dim, num_outputs=num_outputs,
                          trunk_depth=trunk_depth, branch_depth=branch_depth,
                          activation=activation)

        # One zero-initialized, batch-shared full-rank base weight W_{0,l}
        # per conditional hidden layer (same layer indexing as s_w_l/s_b_l).
        self.W0 = nn.ParameterList([
            nn.Parameter(torch.zeros(self.trunk_dims[i + 1], self.trunk_dims[i]))
            for i in range(1, len(self.trunk_dims) - 2)
        ])

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
            s_w_l = params[:, offset:offset + s]
            s_b_l = params[:, offset + s:offset + 2 * s]

            weight1 = self.trunk_net.hidden_0_net[2 * layer_idx].weight   # U_l:   [dims[i+1], s]
            weight2 = self.trunk_net.hidden_1_net[2 * layer_idx].weight  # V_l^T: [s, dims[i]]
            bias_basis = self.trunk_net.bias_basis_net[layer_idx].weight  # B_l:  [dims[i+1], s]

            w_z = torch.einsum("hs,bs->bhs", weight1, s_w_l)
            weight = torch.einsum("bhs,sr->bhr", w_z, weight2)
            weight = weight + self.W0[layer_idx].unsqueeze(0)  # W_{0,l} + U_l diag(s_w_l) V_l^T
            bias = torch.einsum("hs,bs->bh", bias_basis, s_b_l)

            y = torch.einsum("bhr,bnr->bnh", weight, y)
            y = y + bias.unsqueeze(1)
            y = self._trunk_act(y)

        y = self.trunk_net.last_net(y)

        return y
