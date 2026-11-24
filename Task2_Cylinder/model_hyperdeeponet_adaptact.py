"""
Ablation of plain HyperDeepONet (model_hyperdeeponet.py): adds
Fusion_DeepONet-style adaptive activation parameters to the trunk's
activated layers.

    plain HyperDeepONet:  y = act(y)                                     (GELU/Tanh, no learnable params)
    this variant:         y = act(10*a_l*y + c_l) + 10*a1_l*sin(10*F1_l*y + c1_l)

Same adaptive-activation form and init as model_LRNR_adaptact.py /
Fusion_DeepONet: a_l=c_l=F1_l=0.1, a1_l=c1_l=0.0.

HyperDeepONet's own forward already activates exactly `len(trunk_dims)-2
== trunk_depth` layers (every layer except the last, which has no
activation) -- unlike LRNR, where only the first_net + middle conditional
layers are activated by construction. So here all `trunk_depth` activated
layers are branch-generated (no separate "first_net"/"last_net" split to
worry about), one (a_l,c_l,F1_l,a1_l,c1_l) tuple per activated layer.

branch_net, t_para layout, _branch_forward, forward() are all reused
unchanged from HyperDeepONet.
"""

import torch
import torch.nn as nn

from model_hyperdeeponet import HyperDeepONet


class HyperDeepONet_AdaptAct(HyperDeepONet):
    def __init__(self, branch_dim=674, trunk_dim=2, hidden_dim=46,
                 num_outputs=4, trunk_depth=3, branch_depth=3,
                 activation='GELU'):
        super().__init__(branch_dim=branch_dim, trunk_dim=trunk_dim,
                          hidden_dim=hidden_dim, num_outputs=num_outputs,
                          trunk_depth=trunk_depth, branch_depth=branch_depth,
                          activation=activation)

        n_act_layers = len(self.trunk_dims) - 2  # == trunk_depth
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

        y = x_trunk
        start = 0

        for i in range(len(self.trunk_dims) - 2):
            d_in, d_out = self.trunk_dims[i], self.trunk_dims[i + 1]

            w_sz = d_in * d_out
            weight = params[:, start:start + w_sz].reshape(B, d_out, d_in)
            start += w_sz
            bias = params[:, start:start + d_out].reshape(B, 1, d_out)
            start += d_out

            y = torch.einsum("bij,bgj->bgi", weight, y) + bias
            y = self._adaptive_act(y, i)

        # Last layer: no activation
        d_in, d_out = self.trunk_dims[-2], self.trunk_dims[-1]
        w_sz = d_in * d_out
        weight = params[:, start:start + w_sz].reshape(B, d_out, d_in)
        start += w_sz
        bias = params[:, start:start + d_out].reshape(B, 1, d_out)

        y = torch.einsum("bij,bgj->bgi", weight, y) + bias
        return y
