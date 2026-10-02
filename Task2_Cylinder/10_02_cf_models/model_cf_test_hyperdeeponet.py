"""
cf_test_HyperDeepONet: cf_hyperdeeponet (model_cf_hyperdeeponet.py) with the
branch's "read-out" (what becomes the trunk weight) and "carry" (what
continues into the next branch depth) pathways decoupled.

Diagnosed problem in cf_hyperdeeponet: at each branch depth i, the SAME
pre-activation tensor h_lin = Linear_i(hyper_input) is used for two
different jobs at once:
  1. params_list.append(h_lin)             -- read out AS the trunk's
     weight/bias for that depth
  2. hyper_input = act(h_lin)              -- carried forward as the
     branch's own representation into depth i+1

These two jobs want different things from the same numbers (job 1 wants
"a good linear operator value", job 2 wants "a representation that's easy
to keep transforming"), and forcing one tensor to serve both seems to be
why cf_hyperdeeponet (4.6-5.6% Rel L2 @200K) underperforms even plain
HyperDeepONet (1.9% @200K) and c_hyperdeeponet (2.2-3.3% @200K).

Fix here: add one extra nn.Linear "read-out head" per branch depth, applied
to the SAME input the carry-path's Linear_i sees, so the trunk weight comes
from an independent linear map instead of being read off the carry
pathway's own pre-activation output. The carry pathway itself (Linear_i +
activation -> next hyper_input) is left completely untouched, so the
branch's internal representation is free to evolve however is best for
"being a branch", without being pinned to also being a good weight value.

    h_lin  = Linear_i(hyper_input)          # unchanged carry-path linear
    r_i    = ReadOut_i(hyper_input)         # NEW, separate linear, same input
    params_list.append(r_i)                 # r_i is what the trunk sees now
    hyper_input = act(h_lin)                 # carry continues exactly as before

Everything else (latent_chunk mechanism, num_chunks = hidden_dim + 1,
directly-learned first/last trunk layers via _trunk_MLP, trunk_dims,
activation choice, chunk_in/chunk_out convention) is unchanged from
model_cf_hyperdeeponet.py -- this file isolates exactly one variable.
"""

import torch
import torch.nn as nn


class cf_test_HyperDeepONet(nn.Module):
    def __init__(self, branch_dim=674, trunk_dim=2, hidden_dim=46, num_basis=100,
                 num_outputs=4, trunk_depth=3, branch_depth=3,
                 activation='GELU', chunk_in=100, chunk_out=100):
        super().__init__()

        if activation == 'Tanh':
            act = nn.Tanh
            self._trunk_act = torch.tanh
        elif activation == 'GELU':
            act = nn.GELU
            self._trunk_act = nn.functional.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Trunk architecture: [trunk_dim, hidden, ..., hidden, num_outputs].
        if num_basis is None:
            self.trunk_dims = [trunk_dim] + [hidden_dim] * trunk_depth + [num_outputs]
        else:
            self.trunk_dims = [trunk_dim] + [hidden_dim] * trunk_depth + [num_basis, num_outputs]

        self.chunk_in = chunk_in
        self.chunk_out = chunk_out  # vestigial, kept only for CLI/call-site compatibility -- see _MLP fix below
        self.num_chunks = hidden_dim + 1
        self.latent_chunk = nn.Parameter(torch.randn(self.num_chunks, chunk_in))

        # Branch (carry path): single network -> hidden_dim per depth, same
        # as cf_hyperdeeponet EXCEPT the dead trailing layer is gone (see
        # _MLP below): dims now ends at hidden_dim, not chunk_out, since
        # nothing ever read a chunk_out-shaped output here.
        self.branch_dims = [branch_dim + chunk_in] + [hidden_dim] * branch_depth
        self.trunk_dims = [trunk_dim] + [hidden_dim] * branch_depth + [num_outputs]
        self.branch_net = _MLP(self.branch_dims, act)
        self.trunk_net = _trunk_MLP(self.trunk_dims, act)

        # NEW: one separate read-out head per branch depth whose params_list
        # entry is actually consumed by the trunk (i = 1 .. branch_depth-1;
        # depth 0's entry is discarded by _trunk_forward's loop, which starts
        # at i=1 -- exactly as in cf_hyperdeeponet -- so no head is added
        # there; that would just duplicate the biggest branch layer,
        # (branch_dim+chunk_in) -> hidden_dim, for nothing). Each head has
        # the SAME (in_features, out_features) as the carry path's Linear_i.
        self.readout_heads = nn.ModuleDict({
            str(i): nn.Linear(self.branch_dims[i], self.branch_dims[i + 1])
            for i in range(1, len(self.branch_dims) - 1)
        })

    def _branch_forward_part(self, x, i):
        """Run branch net layer i on x (carry path, unchanged)."""
        return self.branch_net.net[i](x)

    def _trunk_forward(self, params, x_trunk):
        """Hypernetwork trunk: params -> weights/biases -> forward pass. Unchanged from cf_hyperdeeponet."""
        B = params[0].shape[0]
        if x_trunk.dim() == 2:
            x_trunk = x_trunk.unsqueeze(0).expand(B, -1, -1)

        _, N, _ = x_trunk.shape
        y = x_trunk  # [B, N, trunk_dim]

        y = self.trunk_net.net[0](y)
        y = self.trunk_net.net[1](y)

        for i in range(1, len(self.trunk_dims) - 2):
            d_in, d_out = self.trunk_dims[i], self.trunk_dims[i + 1]

            weight = params[i][:, :-1, :]
            bias = params[i][:, -1:, :].reshape(B, 1, d_out)

            y = torch.einsum("bij,bgj->bgi", weight, y) + bias
            y = self._trunk_act(y)

        # Last layer: no activation
        y = self.trunk_net.net[-1](y)
        return y

    def forward(self, x_branch, x_trunk):
        """
        Args:
            x_branch: [B, branch_dim]  sensor values
            x_trunk:  [B, N, trunk_dim]  query coordinates

        Returns:
            [B, N, num_outputs]
        """
        B = x_branch.shape[0]
        K = self.num_chunks

        x_branch = x_branch.unsqueeze(1).repeat(1, K, 1)
        z = self.latent_chunk.unsqueeze(0).expand(B, -1, -1)
        hyper_input = torch.cat([x_branch, z], dim=-1)  # [B, K, branch_dim + chunk_in]

        params_list = []

        for i in range(len(self.branch_dims) - 1):
            if str(i) in self.readout_heads:
                # Read-out: separate linear, same input as the carry path's
                # Linear_i, but its own weights -- this is what the trunk sees.
                r_out = self.readout_heads[str(i)](hyper_input)
            else:
                # depth 0: params_list[0] is discarded by _trunk_forward
                # anyway (its loop starts at i=1), so just mirror the old
                # cf_hyperdeeponet behavior here -- no head is allocated.
                r_out = None

            # Carry: exactly as in cf_hyperdeeponet, untouched by read-out.
            h_lin = self._branch_forward_part(hyper_input, 2 * i)
            params_list.append(h_lin if r_out is None else r_out)
            hyper_input = self._branch_forward_part(h_lin, 2 * i + 1)

        return self._trunk_forward(params_list, x_trunk)


class _MLP(nn.Module):
    """Fully-connected stack: Linear -> Act -> ... -> Linear -> Act.

    Every layer here is read by cf_test_HyperDeepONet's forward loop (which
    walks net[2*i]/net[2*i+1] for i in range(len(dims)-1)). Unlike
    cf_hyperdeeponet's _MLP, this does NOT append one extra bare
    Linear(dims[-2], dims[-1]) beyond that -- that trailing layer was never
    read by the forward loop there, silently wasting
    hidden_dim*chunk_out+chunk_out parameters (~16% of the model at the 50K
    tier). `dims` should end at hidden_dim, not chunk_out.
    """
    def __init__(self, dims, act):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(act())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class _trunk_MLP(nn.Module):
    """Simple fully-connected stack: Linear -> Act -> ... -> Linear."""
    def __init__(self, dims, act):
        super().__init__()
        layers = []
        layers.append(nn.Linear(dims[0], dims[1]))
        layers.append(act())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
