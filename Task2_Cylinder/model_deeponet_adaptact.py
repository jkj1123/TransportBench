"""
Ablation of plain DeepONet (model_deeponet.py, BoltzmannDeepONet): adds
Fusion_DeepONet-style adaptive activation parameters to the TRUNK only
(branch stays plain GELU/Tanh, unchanged -- matches the convention used in
model_LRNR_adaptact.py / model_hyperdeeponet_adaptact.py, where only the
trunk gets the adaptive treatment).

    plain DeepONet trunk:  y = act(y)                                    (GELU/Tanh, no learnable params)
    this variant:          y = act(10*a_l*y + c_l) + 10*a1_l*sin(10*F1_l*y + c1_l)

Same adaptive-activation form/init as the other *_adaptact.py files:
a_l=c_l=F1_l=0.1, a1_l=c1_l=0.0.

BoltzmannDeepONet's trunk_net has `depth+1` Linear layers, EVERY one
followed by an activation (unlike LRNR/HyperDeepONet, where the last
layer has no activation) -- so here all `depth+1` layers get their own
adaptive-activation tuple.

branch_net, the branch-trunk dot product combination, and forward()'s
interface are all reused unchanged.
"""

import torch
import torch.nn as nn


class BoltzmannDeepONet_AdaptAct(nn.Module):
    def __init__(self, branch_dim, trunk_dim, hidden_dim=256, num_outputs=4, depth=5, activation='GELU'):
        super().__init__()

        if activation == 'GELU':
            self.act = nn.GELU()
            self._base_act_fn = nn.functional.gelu
        elif activation == 'Tanh':
            self.act = nn.Tanh()
            self._base_act_fn = torch.tanh
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Branch net: unchanged, plain activation (not adaptified).
        branch_layers = []
        branch_layers.extend([nn.Linear(branch_dim, hidden_dim), self.act])
        for _ in range(depth - 1):
            branch_layers.extend([nn.Linear(hidden_dim, hidden_dim), self.act])
        branch_layers.append(nn.Linear(hidden_dim, hidden_dim * num_outputs))
        self.branch_net = nn.Sequential(*branch_layers)

        # Trunk net: same Linear layers as BoltzmannDeepONet, but we apply
        # them one at a time in forward() so we can swap in the adaptive
        # activation instead of the plain self.act after each one.
        self.trunk_linears = nn.ModuleList()
        self.trunk_linears.append(nn.Linear(trunk_dim, hidden_dim))
        for _ in range(depth - 1):
            self.trunk_linears.append(nn.Linear(hidden_dim, hidden_dim))
        self.trunk_linears.append(nn.Linear(hidden_dim, hidden_dim))  # matches the extra final layer

        n_act_layers = len(self.trunk_linears)  # depth + 1, every trunk layer is activated
        self.at = nn.Parameter(torch.full((n_act_layers,), 0.1))
        self.ct = nn.Parameter(torch.full((n_act_layers,), 0.1))
        self.F1t = nn.Parameter(torch.full((n_act_layers,), 0.1))
        self.a1t = nn.Parameter(torch.zeros(n_act_layers))
        self.c1t = nn.Parameter(torch.zeros(n_act_layers))

        self.num_outputs = num_outputs
        self.hidden_dim = hidden_dim

    def _adaptive_act(self, y, layer_idx):
        a, c = self.at[layer_idx], self.ct[layer_idx]
        F1, a1, c1 = self.F1t[layer_idx], self.a1t[layer_idx], self.c1t[layer_idx]
        return self._base_act_fn(10 * a * y + c) + 10 * a1 * torch.sin(10 * F1 * y + c1)

    def forward(self, x_branch, x_trunk):
        B_out = self.branch_net(x_branch)  # [Batch, hidden*num_outputs]

        y = x_trunk  # [N_points, trunk_dim]
        for i, lin in enumerate(self.trunk_linears):
            y = lin(y)
            y = self._adaptive_act(y, i)
        T_out = y  # [N_points, hidden]

        B_out_reshaped = B_out.view(-1, self.num_outputs, self.hidden_dim)
        prediction = torch.einsum("bkh, nh -> bkn", B_out_reshaped, T_out)
        prediction = prediction.permute(0, 2, 1)
        return prediction
