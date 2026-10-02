"""
Fusion_WbWt_DeepONet: SMOKE-TEST variant of Fusion_DeepONet
(model_fusion_deeponet.py) that replaces the "skip[i] * (whole trunk layer
output)" gating with a decomposed weight/bias cross-fusion:

    act(W_b @ x_branch) * act(W_t @ x_trunk)   +   act(b_b) * act(b_t)

instead of

    skip[i] * act(fused branch/trunk formula)

Design (per discussion, "option a" -- minimal change from the original):
- Branch still evolves on its own, independent of trunk, via a plain
  act(Linear(x)) recurrence each layer (no scale-shift/sin -- simplified for
  this smoke test).
- Skip accumulation is kept, but now applies to BOTH decomposed streams
  separately: the per-layer act(W_b@x) term and the per-layer act(b_b) term
  are each accumulated across layers (skip_W[i] = skip_W[i-1] + act(Wb_raw),
  skip_b[i] = skip_b[i-1] + act(bb_raw)), mirroring the original single-
  stream skip[i] = skip[i-1] + skip[i].
- Trunk's per-layer weight/bias are likewise split via F.linear (no bias)
  for the W-part and the bias vector directly for the b-part.
- The final layer (branch_net[-1], trunk_net[-1]) and the closing dot
  product are unchanged from Fusion_DeepONet.

Not yet validated beyond a smoke test -- only enough that a proper
experiment can be designed once we see it actually converges.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Fusion_WbWt_DeepONet(nn.Module):
    def __init__(self, branch_dim=1, trunk_dim=2, hidden_dim=280, num_outputs=10, depth=5, activation='GELU'):
        super().__init__()

        if activation == 'GELU':
            self.act = nn.GELU()
        elif activation == 'Tanh':
            self.act = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.L = depth + 1

        # Branch Net
        self.branch_net = nn.ModuleList()
        self.branch_net.append(nn.Linear(branch_dim, hidden_dim))
        for _ in range(depth - 1):
            self.branch_net.append(nn.Linear(hidden_dim, hidden_dim))
        self.branch_net.append(nn.Linear(hidden_dim, hidden_dim * num_outputs))

        # Trunk Net
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

        skip_W = None  # accumulated act(W_b @ x_branch), shape [B, hidden_dim]
        skip_b = None  # accumulated act(bias_b),          shape [hidden_dim]

        for i in range(self.L - 1):
            layer_b = self.branch_net[i]
            layer_t = self.trunk_net[i]

            # --- branch: W/b decomposition (for the cross-fusion) ---
            Wx_b = F.linear(x_branch, layer_b.weight)      # [B, hidden_dim], no bias
            b_b = layer_b.bias                              # [hidden_dim]

            Wb_part = self.act(Wx_b)                        # [B, hidden_dim]
            bb_part = self.act(b_b)                          # [hidden_dim]

            skip_W = Wb_part if skip_W is None else skip_W + Wb_part
            skip_b = bb_part if skip_b is None else skip_b + bb_part

            # --- branch's own independent progression to the next layer ---
            x_branch = self.act(Wx_b + b_b)                  # plain act(Linear(x))

            # --- trunk: W/b decomposition ---
            Wx_t = F.linear(x_trunk, layer_t.weight)         # [B, N, hidden_dim], no bias
            b_t = layer_t.bias                                # [hidden_dim]

            Wt_part = self.act(Wx_t)                          # [B, N, hidden_dim]
            bt_part = self.act(b_t)                            # [hidden_dim]

            # cross-fusion: broadcast skip_W [B,hidden] over N, skip_b/bt_part [hidden] over B,N
            x_trunk = torch.einsum('bk,bik->bik', skip_W, Wt_part) + (skip_b * bt_part)

        x_branch = self.branch_net[-1](x_branch)
        x_trunk = self.trunk_net[-1](x_trunk)

        B_out_reshaped = x_branch.view(-1, self.num_outputs, self.hidden_dim)
        prediction = torch.einsum("bnk, bik -> bin", B_out_reshaped, x_trunk)
        return prediction
