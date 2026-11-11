"""
HyperDeepONet: DeepONet with a hypernetwork trunk.

Key ideas from Lee & Shin:
- Branch net outputs ARE the trunk net's weights/biases (hypernetwork).
- No learned parameters in the trunk — all trunk params come from the branch output.
"""

import torch
import torch.nn as nn


class LNLR(nn.Module):
    def __init__(self, branch_dim=674, trunk_dim=2, hidden_dim=46,
                 num_outputs=4, trunk_depth=3, branch_depth=3,
                 activation='GELU', s=12, s_w_offset=0.0):
        super().__init__()
        # Fixed constant added to every element of s_w_l before use (does not
        # affect s_b_l). s_w_offset=0.0 (default) reproduces the original
        # behavior exactly. A nonzero offset keeps W_l = U_l diag(s_w_l) V_l^T
        # from collapsing to the all-zero solution the branch can otherwise
        # drive it to, since diag(s_w_l) can no longer be driven to exactly 0.
        self.s_w_offset = s_w_offset

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

        self.s = s
        # One independent (s_w_l, s_b_l) pair of rank-`s` coefficient vectors
        # per hypernetwork-generated hidden layer (trunk_depth - 1 of them).
        self.num_hidden_layers = trunk_depth - 1
        t_para = self.num_hidden_layers * 2 * self.s

        # Branch: single network → t_para (every layer's weight/bias coefficients)
        branch_dims = [branch_dim] + [hidden_dim] * branch_depth + [t_para]
        self.branch_net = _MLP(branch_dims, act)
        self.trunk_net = _MLP_reduced(self.trunk_dims, self.s, act)

        self.num_outputs = num_outputs

    def _branch_forward(self, x):
        """Run branch net on x → trunk parameters."""
        return self.branch_net(x)  # [B, t_para]

    def _trunk_forward(self, params, x_trunk):
        """Hypernetwork trunk: params → per-layer weights/biases → forward pass.

        `params` ([B, t_para]) is sliced into one (s_w_l, s_b_l) pair per
        hypernetwork-generated hidden layer l: W_l = U_l diag(s_w_l) V_l^T,
        b_l = B_l s_b_l, with U_l, V_l, B_l all learned (trunk-resident)
        matrices and s_w_l, s_b_l the per-layer, per-sample coefficient
        vectors coming out of the single branch net.
        """
        B = params.shape[0]  # use branch batch size (handles shared trunk)
        # Normalize to 3D: [B, N, trunk_dim]
        if x_trunk.dim() == 2:
            x_trunk = x_trunk.unsqueeze(0).expand(B, -1, -1)

        _, N, _ = x_trunk.shape
        y = x_trunk  # [B, N, trunk_dim]

        y = self.trunk_net.first_net(y)

        s = self.s
        for i in range(1, len(self.trunk_dims) - 2):
            layer_idx = i - 1
            offset = layer_idx * 2 * s
            s_w_l = params[:, offset:offset + s] + self.s_w_offset
            s_b_l = params[:, offset + s:offset + 2 * s]

            weight1 = self.trunk_net.hidden_0_net[2 * layer_idx].weight   # U_l:   [dims[i+1], s]
            weight2 = self.trunk_net.hidden_1_net[2 * layer_idx].weight  # V_l^T: [s, dims[i]]
            bias_basis = self.trunk_net.bias_basis_net[layer_idx].weight  # B_l:  [dims[i+1], s]

            w_z = torch.einsum("hs,bs->bhs", weight1, s_w_l)
            weight = torch.einsum("bhs,sr->bhr", w_z, weight2)
            bias = torch.einsum("hs,bs->bh", bias_basis, s_b_l)

            y = torch.einsum("bhr,bnr->bnh", weight, y)
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
        params = self._branch_forward(x_branch)  # [B, t_para]
        return self._trunk_forward(params, x_trunk)


class _MLP(nn.Module):
    """Simple fully-connected stack: Linear → Act → ... → Linear."""
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
    """Simple fully-connected stack: Linear → Act → ... → Linear.

    hidden_0_net / hidden_1_net hold each layer's U_l / V_l^T (low-rank
    weight factors); bias_basis_net holds each layer's B_l (bias basis
    matrix). Only the raw `.weight` of these Linear modules is used (see
    LNLR._trunk_forward), so the Act() modules interleaved in
    hidden_0_net/hidden_1_net below are never invoked -- unchanged from the
    original implementation.
    """
    def __init__(self, dims, diag ,act):
        super().__init__()
        first_layers = []
        hidden_layers_0 = []
        hidden_layers_1 = []
        bias_basis_layers = []
        last_layers = []

        first_layers.append(nn.Linear(dims[0], dims[1]))
        first_layers.append(act())

        for i in range(1, len(dims) - 2):
            # bias=False: only .weight (U_l) is ever read (see LNLR._trunk_forward);
            # the per-layer bias now comes from bias_basis_net (B_l) instead.
            hidden_layers_0.append(nn.Linear(diag, dims[i + 1], bias=False))
            hidden_layers_0.append(act())
            hidden_layers_1.append(nn.Linear(dims[i], diag, bias = False))
            hidden_layers_1.append(act())
            bias_basis_layers.append(nn.Linear(diag, dims[i + 1], bias=False))
        last_layers.append(nn.Linear(dims[-2], dims[-1]))
        self.first_net = nn.Sequential(*first_layers)
        self.hidden_0_net = nn.Sequential(*hidden_layers_0)
        self.hidden_1_net = nn.Sequential(*hidden_layers_1)
        self.bias_basis_net = nn.ModuleList(bias_basis_layers)
        self.last_net = nn.Sequential(*last_layers)
