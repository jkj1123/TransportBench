"""
cf_test2_HyperDeepONet: cf_test_hyperdeeponet (dead-layer fixed, read/carry
decoupled) combined with three further changes (D + B + A, in the order
they were proposed):

D. Chunk-for-low-rank instead of chunk-for-full-matrix.
   cf_test/cf_hyperdeeponet used K = hidden_dim + 1 chunk positions to
   generate one FULL [d_in, d_out] weight matrix row-by-row (plus a bias
   row) -- exactly the mechanism that has never beaten model_L_hyperdeeponet
   (plain low-rank U(b)V(b)^T, no chunking) at any matched param budget this
   session (1.7-1.9% vs 2.2-4.3%). Here the chunk/latent-code trick is
   reused for something it's actually well suited to: generating the
   LOW-RANK factors themselves. K is now r (rank, default 8), and each of
   the r chunk positions' branch output is split into one column of U_l(b)
   ([d_out]) and one row of V_l(b)^T ([d_in]) for each middle trunk layer l:

       W_l(b) @ y = U_l(b) @ (V_l(b)^T @ y)      (same convention/einsums
                                                    as model_L_hyperdeeponet.py)

   No shared W0 backbone is added (that was the separate L_D/LDg ablation,
   not part of this change) -- U_l(b), V_l(b) are the entire condition-
   dependent weight, exactly like the plain Low-Rank HyperDeepONet, just
   generated via the chunk/latent mechanism instead of one big linear head.
   Bias is a plain shared nn.Parameter per middle layer (not branch-
   generated) to keep the branch's output budget focused on U/V.

B. Re-inject the raw physical condition (Kn, Ma = x_branch) at every
   middle-layer read-out, not just once at the very first concatenation.
   Each read-out head sees [carry-path hyper_input, raw x_branch] so the
   physical condition doesn't have to survive un-degraded through several
   branch depths of its own transformation before it reaches a given
   layer's weight generation.

A. Normalize the branch's hypernetwork input. `latent_chunk` was
   `torch.randn(K, chunk_in)` with no scale control, feeding straight into
   the first branch Linear alongside x_branch (branch_dim=2, tiny by
   comparison) -- suspected to swamp the physical signal. Fixed by (1)
   initializing latent_chunk at 0.1x scale and (2) LayerNorm on the
   [x_branch, latent] concatenation before it enters branch_net.

The dead trailing-Linear bug from cf_hyperdeeponet and the read/carry
decoupling from cf_test_hyperdeeponet are both carried over unchanged
here (see model_cf_test_hyperdeeponet.py for that reasoning) -- this file
adds exactly D, B, A on top.
"""

import torch
import torch.nn as nn


class cf_test2_HyperDeepONet(nn.Module):
    def __init__(self, branch_dim=674, trunk_dim=2, hidden_dim=46,
                 num_outputs=4, trunk_depth=3, branch_depth=3,
                 activation='GELU', chunk_in=100, r=8):
        super().__init__()

        if activation == 'Tanh':
            act = nn.Tanh
            self._trunk_act = torch.tanh
        elif activation == 'GELU':
            act = nn.GELU
            self._trunk_act = nn.functional.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        assert branch_depth == trunk_depth, (
            "cf_test2's carry loop walks branch_depth steps and reads out "
            "at the same depth indices the trunk's middle layers use -- "
            "keep these equal, exactly as cf_hyperdeeponet/cf_test did."
        )

        self.trunk_dims = [trunk_dim] + [hidden_dim] * trunk_depth + [num_outputs]
        self.branch_dim = branch_dim
        self.chunk_in = chunk_in
        self.r = r

        # First/last trunk layers: directly learned, shared (unchanged from
        # cf_hyperdeeponet/cf_test's philosophy).
        self.trunk_in = nn.Linear(trunk_dim, hidden_dim)
        self.trunk_out = nn.Linear(hidden_dim, num_outputs)

        # Middle trunk layers: low-rank, condition-dependent, chunk-generated.
        self.n_middle = len(self.trunk_dims) - 3  # matches L_hyperdeeponet's convention
        self.middle_depths = list(range(1, 1 + self.n_middle))  # branch-depth indices that feed a middle layer
        self.bias_list = nn.ParameterList([
            nn.Parameter(torch.zeros(hidden_dim)) for _ in range(self.n_middle)
        ])

        # (A) small-scale init + LayerNorm on the branch's hypernetwork input
        self.latent_chunk = nn.Parameter(torch.randn(r, chunk_in) * 0.1)
        self.input_norm = nn.LayerNorm(branch_dim + chunk_in)

        # Carry path: plain deep net over the [B, r, ...] chunk batch,
        # UNTOUCHED by read-out (same decoupling as cf_test).
        self.branch_dims = [branch_dim + chunk_in] + [hidden_dim] * branch_depth
        self.branch_net = _MLP(self.branch_dims, act)

        # (D) + (B): one read-out head per middle depth, each producing
        # [U-column (d_out) ; V-row (d_in)] = 2*hidden_dim values per chunk
        # position, from [carry hyper_input ; raw x_branch] (re-injected).
        self.readout_heads = nn.ModuleDict({
            str(i): nn.Linear(self.branch_dims[i] + branch_dim, 2 * hidden_dim)
            for i in self.middle_depths
        })

        self.hidden_dim = hidden_dim
        self.num_outputs = num_outputs

    def forward(self, x_branch, x_trunk):
        """
        Args:
            x_branch: [B, branch_dim]  sensor values (Kn, Ma)
            x_trunk:  [N, trunk_dim] or [B, N, trunk_dim]  query coordinates

        Returns:
            [B, N, num_outputs]
        """
        B = x_branch.shape[0]
        r = self.r

        if x_trunk.dim() == 2:
            x_trunk = x_trunk.unsqueeze(0).expand(B, -1, -1)
        _, N, _ = x_trunk.shape

        # --- Branch / hypernetwork side: generate U_l(b), V_l(b) per middle layer.
        x_branch_rep = x_branch.unsqueeze(1).expand(B, r, self.branch_dim)  # (B) re-injected raw condition
        z = self.latent_chunk.unsqueeze(0).expand(B, r, self.chunk_in)
        hyper_input = torch.cat([x_branch_rep, z], dim=-1)
        hyper_input = self.input_norm(hyper_input)  # (A)

        low_rank = {}  # middle_depth -> (U [B,d_out,r], V^T [B,r,d_in])
        for i in range(len(self.branch_dims) - 1):
            if i in self.middle_depths:
                readout_in = torch.cat([hyper_input, x_branch_rep], dim=-1)  # (B)
                r_out = self.readout_heads[str(i)](readout_in)  # [B, r, 2*hidden_dim]
                U = r_out[..., :self.hidden_dim].transpose(1, 2)  # [B, d_out, r]
                Vt = r_out[..., self.hidden_dim:]                  # [B, r, d_in]  (already V^T convention)
                low_rank[i] = (U, Vt)

            # Carry: unchanged, untouched by read-out.
            h_lin = self.branch_net.net[2 * i](hyper_input)
            hyper_input = self.branch_net.net[2 * i + 1](h_lin)

        # --- Trunk side: first layer (direct), middle layers (low-rank
        # generated, applied without ever materializing the full matrix),
        # last layer (direct) -- same structure as model_L_hyperdeeponet.py.
        y = self.trunk_in(x_trunk)
        y = self._trunk_act(y)

        for idx, i in enumerate(self.middle_depths):
            U, Vt = low_rank[i]
            bias = self.bias_list[idx]
            z_ = torch.einsum("brj,bgj->bgr", Vt, y)      # V(b)^T @ y, [B, N, r]
            y = torch.einsum("bir,bgr->bgi", U, z_) + bias  # U(b) @ z,  [B, N, d_out]
            y = self._trunk_act(y)

        y = self.trunk_out(y)
        return y


class _MLP(nn.Module):
    """Fully-connected stack: Linear -> Act -> ... -> Linear -> Act (no dead trailing layer)."""
    def __init__(self, dims, act):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(act())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
