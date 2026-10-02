"""
LDg_HyperDeepONet: scalar-modulated low-rank conditional trunk.

Drastically reduces the dimensionality of condition-dependent weight
generation compared to both the original Low-Rank HyperDeepONet
(model_L_hyperdeeponet.py, W_l(b) = U_l(b)V_l(b)^T, thousands of
branch-generated values per layer) and L_D_HyperDeepONet
(model_L_D_hyperdeeponet.py, W_l(b) = W0_l + a*U_l(b)V_l(b)^T, still
thousands of branch-generated values per layer).

Here, for each applicable (middle) trunk layer l:

    W_l(b) = W_{0,l} + s_l(b) * U_l V_l^T = W_{0,l} + s_l(b) * D_l

W_{0,l} (full-rank backbone), U_l, V_l (the single learned low-rank
"direction" D_l = U_l V_l^T), AND that layer's bias are ALL ordinary
trainable nn.Parameters -- NONE of them depend on the branch input b.
The branch/hypernetwork generates ONLY one scalar s_l(b) per applicable
layer (unconstrained sign/magnitude, no sigmoid/tanh). For L such layers
the branch's conditional output is just L numbers, not the thousands
needed to generate U_l(b), V_l(b) directly.

Design decisions made where the spec was ambiguous (documented so they're
easy to revisit):
- First and last trunk layers are NOT "applicable" layers here -- they
  keep the same full-rank, branch-generated weight+bias as in
  model_L_hyperdeeponet.py / model_L_D_hyperdeeponet.py, unchanged. Only
  the middle layers (the ones that were low-rank U(b)V(b)^T before) are
  replaced by the W0 + s(b)*UV^T mechanism.
- Bias for the middle layers is likewise made an ordinary shared
  nn.Parameter (not branch-generated), consistent with the instruction
  that the branch should output ONLY the scalars s_1(b)..s_L(b) for the
  applicable layers (previously bias was branch-generated, like
  U(b)/V(b); now everything about the middle layer's affine map is
  shared except the scalar modulation).
- Rank r kept at 8 (same as the existing Low-Rank HyperDeepONet).
- U_l, V_l, W0_l initialized with the same scheme PyTorch's nn.Linear
  uses by default (kaiming_uniform_, a=sqrt(5)); the shared bias uses
  nn.Linear's default bias init (uniform(-1/sqrt(fan_in), 1/sqrt(fan_in))).

Application to trunk activations (same convention as the existing files):

    W_l(b) @ y = W0_l @ y + s_l(b) * (D_l @ y)

D_l = U_l @ V_l^T is recomputed each forward pass (cheap: d_out x r x d_in
matmul, r=8) since U_l, V_l are being trained.
"""

import math
import torch
import torch.nn as nn


class HyperDeepONet(nn.Module):
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

        # Trunk architecture: [trunk_dim, hidden, ..., hidden, num_outputs]
        self.trunk_dims = [trunk_dim] + [hidden_dim] * trunk_depth + [num_outputs]

        r = 8  # same rank as the existing Low-Rank HyperDeepONet
        self.r = r

        # --- Branch (hypernetwork) output width: first/last layer stay
        # full-rank + bias (branch-generated, unchanged); each middle
        # layer now contributes only ONE scalar s_l(b) -- no U(b), V(b),
        # no bias generated for those layers anymore.
        t_para = 0
        for i in range(1):
            t_para += self.trunk_dims[i] * self.trunk_dims[i + 1] + self.trunk_dims[i + 1]

        n_middle = len(self.trunk_dims) - 3  # number of applicable (scalar-modulated) layers
        t_para += n_middle * 1  # exactly one scalar s_l(b) per applicable layer

        for i in range(len(self.trunk_dims) - 2, len(self.trunk_dims) - 1):
            t_para += self.trunk_dims[i] * self.trunk_dims[i + 1] + self.trunk_dims[i + 1]

        self.t_para = t_para
        self.n_middle = n_middle

        # Branch: single network -> t_para (first/last full-rank weights+bias, plus L scalars)
        branch_dims = [branch_dim] + [hidden_dim] * branch_depth + [t_para]
        self.branch_net = _MLP(branch_dims, act)

        # --- Shared, directly-learned parameters for each middle layer:
        # W0_l (full-rank backbone), U_l, V_l (low-rank direction D_l =
        # U_l V_l^T), and bias_l. None of these depend on the branch input.
        self.W0_list = nn.ParameterList()
        self.U_list = nn.ParameterList()
        self.V_list = nn.ParameterList()
        self.bias_list = nn.ParameterList()
        for i in range(1, len(self.trunk_dims) - 2):
            d_in = self.trunk_dims[i]
            d_out = self.trunk_dims[i + 1]

            w0 = nn.Parameter(torch.empty(d_out, d_in))
            nn.init.kaiming_uniform_(w0, a=math.sqrt(5))
            self.W0_list.append(w0)

            u = nn.Parameter(torch.empty(d_out, r))
            nn.init.kaiming_uniform_(u, a=math.sqrt(5))
            self.U_list.append(u)

            v = nn.Parameter(torch.empty(d_in, r))
            nn.init.kaiming_uniform_(v, a=math.sqrt(5))
            self.V_list.append(v)

            bias = nn.Parameter(torch.empty(d_out))
            fan_in = d_in
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(bias, -bound, bound)
            self.bias_list.append(bias)

        self.num_outputs = num_outputs

    def _branch_forward(self, x):
        """Run branch net on x -> [first-layer full-rank, s_1..s_L, last-layer full-rank]."""
        return self.branch_net(x)  # [B, t_para]

    def _trunk_forward(self, params, x_trunk):
        """Hypernetwork trunk: params -> weights/biases/scalars -> forward pass."""
        B = params.shape[0]
        if x_trunk.dim() == 2:
            x_trunk = x_trunk.unsqueeze(0).expand(B, -1, -1)

        _, N, _ = x_trunk.shape
        y = x_trunk  # [B, N, trunk_dim]
        start = 0

        # First layer: full-rank, branch-generated (unchanged)
        for i in range(1):
            d_in, d_out = self.trunk_dims[i], self.trunk_dims[i + 1]
            w_sz = d_in * d_out
            weight = params[:, start:start + w_sz].reshape(B, d_out, d_in)
            start += w_sz
            bias = params[:, start:start + d_out].reshape(B, 1, d_out)
            start += d_out

            y = torch.einsum("bij,bgj->bgi", weight, y) + bias
            y = self._trunk_act(y)

        # Middle layers: W_l(b) = W0_l + s_l(b) * (U_l V_l^T), W0/U/V/bias shared
        for idx, i in enumerate(range(1, len(self.trunk_dims) - 2)):
            s = params[:, start:start + 1]  # [B, 1] -- the ONLY conditional value for this layer
            start += 1

            W0 = self.W0_list[idx]      # [d_out, d_in], shared
            U = self.U_list[idx]        # [d_out, r], shared
            V = self.V_list[idx]        # [d_in, r], shared
            bias = self.bias_list[idx]  # [d_out], shared

            D = U @ V.t()  # [d_out, d_in], recomputed each forward (U, V are being trained)

            y_shared = torch.einsum("ij,bgj->bgi", W0, y)   # W0 @ y,       [B, N, d_out]
            y_dir = torch.einsum("ij,bgj->bgi", D, y)        # D @ y,        [B, N, d_out]
            s_b = s.view(B, 1, 1)                             # [B, 1, 1] for broadcasting

            y = y_shared + s_b * y_dir + bias.view(1, 1, -1)
            y = self._trunk_act(y)

        # Last layer: full-rank, branch-generated, no activation (unchanged)
        d_in, d_out = self.trunk_dims[-2], self.trunk_dims[-1]
        w_sz = d_in * d_out
        weight = params[:, start:start + w_sz].reshape(B, d_out, d_in)
        start += w_sz
        bias = params[:, start:start + d_out].reshape(B, 1, d_out)
        start += d_out

        y = torch.einsum("bij,bgj->bgi", weight, y) + bias  # [B, N, num_outputs]
        return y

    def forward(self, x_branch, x_trunk):
        """
        Args:
            x_branch: [B, branch_dim]  sensor values
            x_trunk:  [N, trunk_dim] or [B, N, trunk_dim]  query coordinates

        Returns:
            [B, N, num_outputs]
        """
        params = self._branch_forward(x_branch)  # [B, t_para]
        return self._trunk_forward(params, x_trunk)

    @torch.no_grad()
    def get_diagnostics(self, x_branch):
        """
        Diagnostic-only utility (does NOT affect training/forward): for a
        batch of branch inputs, returns per-middle-layer:
          ||W0_l||_F, ||D_l||_F, mean/std of s_l(b) over the batch, and
          E_b[|s_l(b)|] * ||D_l||_F / ||W0_l||_F (relative magnitude of the
          conditional update vs. the shared backbone).
        Call this whenever you want to inspect these values; it is not
        wired into the training loop.
        """
        params = self._branch_forward(x_branch)
        B = params.shape[0]
        start = 0
        d_in, d_out = self.trunk_dims[0], self.trunk_dims[1]
        start += d_in * d_out + d_out

        results = []
        for idx in range(self.n_middle):
            s = params[:, start:start + 1].squeeze(-1)  # [B]
            start += 1

            W0 = self.W0_list[idx]
            U = self.U_list[idx]
            V = self.V_list[idx]
            D = U @ V.t()

            w0_norm = W0.norm().item()
            d_norm = D.norm().item()
            s_mean = s.mean().item()
            s_std = s.std().item()
            s_abs_mean = s.abs().mean().item()
            rel_mag = (s_abs_mean * d_norm) / (w0_norm + 1e-12)

            results.append(dict(layer=idx, W0_norm=w0_norm, D_norm=d_norm,
                                 s_mean=s_mean, s_std=s_std, s_abs_mean=s_abs_mean,
                                 rel_magnitude=rel_mag))
        return results


class _MLP(nn.Module):
    """Simple fully-connected stack: Linear -> Act -> ... -> Linear."""
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
