"""
Ablation of LNLR (model_LNLR.py): removes V_l entirely and sets the rank
parameter s to the full hidden width, so each conditional hidden layer's
weight collapses from the low-rank factorization

    W_l(mu) = U_l diag(s_w_l(mu)) V_l^T

to a single fixed matrix U_l with the branch output directly gating y
channel-wise before it:

    y_out = U_l @ (s_w_l(mu) * y_in)   i.e.   W_l(mu) = U_l diag(s_w_l(mu))

Since s == hidden_dim, no projection into a rank-s bottleneck is needed
(s_w_l already has one entry per input channel of y), so V_l is simply
never created. This also means the per-sample weight matrix is never
materialized: y is scaled per-sample first, then passed through the single
SHARED matrix U_l, unlike LNLR which reconstructs a full [B,hidden,hidden]
weight per sample via U_l diag(s_w_l) V_l^T.

bias (b_l = B_l s_b_l), branch net, activation, and first/last trunk layers
are unchanged from model_LNLR.py. model_LNLR.py itself is not modified.
"""

import torch
import torch.nn as nn

from model_LNLR import _MLP


class LNLR_UdiagS(nn.Module):
    def __init__(self, branch_dim=674, trunk_dim=2, hidden_dim=46,
                 num_outputs=4, trunk_depth=3, branch_depth=3,
                 activation='GELU'):
        super().__init__()

        if activation == 'Tanh':
            act = nn.Tanh
            self._trunk_act = torch.tanh
        elif activation == 'GELU':
            act = nn.GELU
            self._trunk_act = nn.functional.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.trunk_dims = [trunk_dim] + [hidden_dim] * trunk_depth + [num_outputs]

        # s is forced to the full hidden width: no low-rank bottleneck, and
        # V_l is unnecessary (s_w_l already matches y_in's channel count).
        self.s = hidden_dim
        self.num_hidden_layers = trunk_depth - 1
        t_para = self.num_hidden_layers * 2 * self.s

        branch_dims = [branch_dim] + [hidden_dim] * branch_depth + [t_para]
        self.branch_net = _MLP(branch_dims, act)
        self.trunk_net = _MLP_UdiagS(self.trunk_dims, self.s, act)

        self.num_outputs = num_outputs

    def _branch_forward(self, x):
        return self.branch_net(x)  # [B, t_para]

    def _trunk_forward(self, params, x_trunk):
        """W_l(mu) = U_l diag(s_w_l(mu)), b_l = B_l s_b_l. U_l, B_l are
        learned trunk-resident matrices (shared across the batch); s_w_l,
        s_b_l are the per-layer, per-sample coefficient vectors coming out
        of the single branch net."""
        B = params.shape[0]
        if x_trunk.dim() == 2:
            x_trunk = x_trunk.unsqueeze(0).expand(B, -1, -1)

        y = self.trunk_net.first_net(x_trunk)  # [B, N, hidden]

        s = self.s
        for i in range(1, len(self.trunk_dims) - 2):
            layer_idx = i - 1
            offset = layer_idx * 2 * s
            s_w_l = params[:, offset:offset + s]            # [B, dims[i]] (dims[i] == s == hidden_dim)
            s_b_l = params[:, offset + s:offset + 2 * s]     # [B, s]

            U_l = self.trunk_net.hidden_0_net[layer_idx].weight          # [dims[i+1], dims[i]]
            bias_basis = self.trunk_net.bias_basis_net[layer_idx].weight  # [dims[i+1], s]

            y_scaled = y * s_w_l.unsqueeze(1)                 # [B, N, dims[i]], per-sample channel gating
            y = torch.einsum("hr,bnr->bnh", U_l, y_scaled)    # shared U_l -- no per-sample weight materialized
            bias = torch.einsum("hs,bs->bh", bias_basis, s_b_l)
            y = y + bias.unsqueeze(1)
            y = self._trunk_act(y)

        y = self.trunk_net.last_net(y)

        return y

    def forward(self, x_branch, x_trunk):
        """
        Args:
            x_branch: [B, branch_dim]  sensor values
            x_trunk:  [N, trunk_dim] or [B, N, trunk_dim]  query coordinates

        Returns:
            [B, N, num_outputs]
        """
        params = self._branch_forward(x_branch)
        return self._trunk_forward(params, x_trunk)


class _MLP_UdiagS(nn.Module):
    """first_net -> per-layer (U_l, B_l) -> last_net. No V_l (see module docstring)."""
    def __init__(self, dims, diag, act):
        super().__init__()
        self.first_net = nn.Sequential(nn.Linear(dims[0], dims[1]), act())

        hidden_0_layers = []
        bias_basis_layers = []
        for i in range(1, len(dims) - 2):
            hidden_0_layers.append(nn.Linear(diag, dims[i + 1], bias=False))    # U_l: [dims[i+1], diag]
            bias_basis_layers.append(nn.Linear(diag, dims[i + 1], bias=False))  # B_l: [dims[i+1], diag]
        self.hidden_0_net = nn.ModuleList(hidden_0_layers)
        self.bias_basis_net = nn.ModuleList(bias_basis_layers)

        self.last_net = nn.Sequential(nn.Linear(dims[-2], dims[-1]))
