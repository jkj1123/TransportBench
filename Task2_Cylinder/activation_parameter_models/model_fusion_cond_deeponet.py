"""
Branch-Conditioned Adaptive Activation Fusion_DeepONet: extends
model_fusion_deeponet.py (not modified by this file) by making the
TrunkNet's adaptive-activation parameters (at, ct, a1t, F1t, c1t) depend on
the branch input u = x_branch, instead of being static layerwise scalars
shared by every sample.

The TrunkNet activation's FORM is kept byte-for-byte identical to the
original:

    f_l(z; u) = act(10*at_l(u)*z + ct_l(u))
                + 10*a1t_l(u) * sin(10*F1t_l(u)*z + c1t_l(u))

Each conditional parameter is a learnable base plus a bounded,
branch-conditioned correction:

    p_l(u) = p_l^base + epsilon * tanh(Delta_p_l(u))        -- epsilon=0.1
                                                                 by default,
                                                                 so the
                                                                 correction
                                                                 is always
                                                                 confined to
                                                                 (-epsilon,
                                                                 epsilon)
                                                                 regardless
                                                                 of u.

p_l^base is a learnable nn.Parameter initialized exactly as in the original
model (at=0.1, ct=0.1, a1t=0.0, F1t=0.1, c1t=0.0). All (Delta_at, Delta_ct,
Delta_a1t, Delta_F1t, Delta_c1t) for every TrunkNet hidden layer come from a
single Conditioning MLP applied once to the ORIGINAL branch input (before
the BranchNet loop below overwrites x_branch):

    self.condition_net = nn.Sequential(
        nn.Linear(branch_dim, 32), nn.GELU(), nn.Linear(32, depth * 5))

Its last Linear is zero-initialized, so at init every correction is exactly
0 and this model is numerically identical to the original Fusion_DeepONet.
BatchNorm is not used anywhere in condition_net.

Only the TrunkNet's adaptive activation is conditioned. The BranchNet's
adaptive activation (ab, cb, a1b, F1b, c1b) stays the original static
per-layer parameters. The skip accumulation, the TrunkNet's fusion einsum,
the last (un-activated) Branch/Trunk Linear, and the final DeepONet inner
product are all unchanged.

Parameter names (`branch_net`, `trunk_net`, `ab`/`cb`/`a1b`/`F1b`/`c1b`, and
`at`/`ct`/`a1t`/`F1t`/`c1t`) are kept identical to model_fusion_deeponet.py,
so an original checkpoint's state_dict loads into this model with
`strict=False` (only this model's new `condition_net.*` keys are missing);
see `load_original_checkpoint()` below for a small convenience wrapper.
"""
import torch
import torch.nn as nn


class sin_act(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class Fusion_DeepONet(nn.Module):
    def __init__(self, branch_dim=1, trunk_dim=2, hidden_dim=280, num_outputs=10, depth=5,
                 activation='GELU', epsilon=0.1, cond_hidden_dim=32):
        """
        Same constructor signature as model_fusion_deeponet.Fusion_DeepONet,
        plus two new optional kwargs:
            epsilon: bound on the branch-conditioned correction to each
                TrunkNet activation parameter (default 0.1, configurable).
            cond_hidden_dim: hidden width of condition_net (default 32, per
                spec: Linear(branch_dim, 32) -> GELU -> Linear(32, depth*5)).
        """
        super().__init__()

        if activation == 'GELU':
            self.act = nn.GELU()
        elif activation == 'Tanh':
            self.act = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.act2 = sin_act()

        self.depth = depth
        self.epsilon = epsilon

        # Fusion-deeponet additional parameters (unchanged init/semantics)
        self.L = depth + 1

        # BranchNet adaptive activation: static, per-layer, unconditioned (unchanged).
        self.ab  = nn.Parameter(torch.full((self.L,), 0.1))
        self.cb  = nn.Parameter(torch.full((self.L,), 0.1))
        self.a1b = nn.Parameter(torch.zeros(self.L,))
        self.F1b = nn.Parameter(torch.full((self.L,), 0.1))
        self.c1b = nn.Parameter(torch.zeros(self.L,))

        # TrunkNet adaptive activation: learnable BASE parameters (this
        # model's only change is adding a branch-conditioned correction on
        # top of these, applied in forward()).
        self.at  = nn.Parameter(torch.full((self.L,), 0.1))
        self.ct  = nn.Parameter(torch.full((self.L,), 0.1))
        self.a1t = nn.Parameter(torch.zeros(self.L,))
        self.F1t = nn.Parameter(torch.full((self.L,), 0.1))
        self.c1t = nn.Parameter(torch.zeros(self.L,))

        # Conditioning MLP: u=x_branch -> per-layer (Delta_at, Delta_ct,
        # Delta_a1t, Delta_F1t, Delta_c1t) for the TrunkNet's activation.
        # Last Linear zero-initialized so the model starts identical to the
        # original (unconditioned) Fusion_DeepONet.
        self.condition_net = nn.Sequential(
            nn.Linear(branch_dim, cond_hidden_dim),
            nn.GELU(),
            nn.Linear(cond_hidden_dim, depth * 5),
        )
        nn.init.zeros_(self.condition_net[-1].weight)
        nn.init.zeros_(self.condition_net[-1].bias)

        # Branch Net
        self.branch_net = nn.ModuleList()
        self.branch_net.append(nn.Linear(branch_dim, hidden_dim))
        for _ in range(depth - 1):
            self.branch_net.append(nn.Linear(hidden_dim, hidden_dim))
        self.branch_net.append(nn.Linear(hidden_dim, hidden_dim * num_outputs))

        # Trunk Net: [Linear1, Linear2, ..., Last Linear]
        self.trunk_net = nn.ModuleList()
        self.trunk_net.append(nn.Linear(trunk_dim, hidden_dim))
        for _ in range(depth - 1):
            self.trunk_net.append(nn.Linear(hidden_dim, hidden_dim))
        self.trunk_net.append(nn.Linear(hidden_dim, hidden_dim))

        self.num_outputs = num_outputs
        self.hidden_dim = hidden_dim

    def _conditional_trunk_params(self, cond):
        """cond: [B, depth, 5] -> 5 tensors, each [B, depth, 1, 1], of the
        ACTUAL (base + epsilon*tanh(correction)) TrunkNet activation
        parameters (not the raw corrections)."""
        # Only the first `depth` of each (L = depth+1) base-parameter array
        # are ever indexed by the trunk activation loop below (i in
        # range(self.L-1) == range(depth)); the last entry (index depth) is
        # allocated like in the original model but never used there either.
        at_cond   = self.at[:self.depth].view(1, -1)   + self.epsilon * torch.tanh(cond[:, :, 0])
        ct_cond   = self.ct[:self.depth].view(1, -1)   + self.epsilon * torch.tanh(cond[:, :, 1])
        a1t_cond  = self.a1t[:self.depth].view(1, -1)  + self.epsilon * torch.tanh(cond[:, :, 2])
        F1t_cond  = self.F1t[:self.depth].view(1, -1)  + self.epsilon * torch.tanh(cond[:, :, 3])
        c1t_cond  = self.c1t[:self.depth].view(1, -1)  + self.epsilon * torch.tanh(cond[:, :, 4])
        return at_cond, ct_cond, a1t_cond, F1t_cond, c1t_cond

    def get_layerwise_activation_params(self, x_branch):
        """Convenience method: for a batch of branch inputs u=(Ma, Kn), ...,
        returns the actual per-layer TrunkNet activation parameters that
        forward() would use for each sample.

        Args:
            x_branch: [B, branch_dim]

        Returns:
            dict with keys 'at','ct','a1t','F1t','c1t', each a [B, depth]
            tensor (one value per sample per conditioned hidden layer).
        """
        with torch.no_grad():
            cond = self.condition_net(x_branch).view(-1, self.depth, 5)
            at_c, ct_c, a1t_c, F1t_c, c1t_c = self._conditional_trunk_params(cond)
        return {'at': at_c, 'ct': ct_c, 'a1t': a1t_c, 'F1t': F1t_c, 'c1t': c1t_c}

    def forward(self, x_branch, x_trunk):

        """Forward pass.

        Args:
            x_branch: [Batch, branch_dim]
            x_trunk:  [N_points, trunk_dim] or [Batch, N_points, trunk_dim]

        Returns:
            [Batch, N_points, num_outputs]
        """

        if x_trunk.dim() == 2:
            # [N, trunk_dim] -> [1, N, trunk_dim] -> [B, N, trunk_dim]
            x_trunk = x_trunk.unsqueeze(0).repeat(x_branch.shape[0], 1, 1)

        # Original branch input (Ma, Kn), BEFORE the BranchNet loop below
        # overwrites x_branch -- used only to condition the TrunkNet's
        # adaptive activation.
        x_branch_orig = x_branch

        cond = self.condition_net(x_branch_orig)        # [B, depth*5]
        cond = cond.view(-1, self.depth, 5)              # [B, depth, 5]
        at_cond, ct_cond, a1t_cond, F1t_cond, c1t_cond = self._conditional_trunk_params(cond)
        # Reshape each layer's per-sample scalar to [B, 1, 1] for broadcasting
        # against trunk_lin: [B, N, hidden_dim].

        skip = []

        for i in range(self.L-1):

            branch_lin = self.branch_net[i](x_branch)
            x_branch = self.act(10*self.ab[i]*branch_lin+self.cb[i])+\
            10*self.a1b[i]*self.act2(10*self.F1b[i]*branch_lin+self.c1b[i])
            skip.append(x_branch)

        for i in range(1,self.L-1):
            skip[i] = skip[i-1]+skip[i]

        for i in range(self.L-1):

            trunk_lin = self.trunk_net[i](x_trunk)

            at_i = at_cond[:, i].view(-1, 1, 1)
            ct_i = ct_cond[:, i].view(-1, 1, 1)
            a1t_i = a1t_cond[:, i].view(-1, 1, 1)
            F1t_i = F1t_cond[:, i].view(-1, 1, 1)
            c1t_i = c1t_cond[:, i].view(-1, 1, 1)

            x_trunk = self.act(10*at_i*trunk_lin+ct_i)+\
            10*a1t_i*self.act2(10*F1t_i*trunk_lin+c1t_i)

            x_trunk = torch.einsum('bk,bik->bik', skip[i], x_trunk)

        x_branch = self.branch_net[-1](x_branch)
        x_trunk = self.trunk_net[-1](x_trunk)

        B_out_reshaped = x_branch.view(-1, self.num_outputs, self.hidden_dim)
        # [Batch, num_outputs, hidden_dim]

        prediction = torch.einsum("bnk, bik -> bin", B_out_reshaped, x_trunk)
        # [Batch, N_points, num_outputs]

        return prediction

    def load_original_checkpoint(self, path, map_location='cpu'):
        """Load a model_fusion_deeponet.Fusion_DeepONet checkpoint's common
        weights (branch_net, trunk_net, ab/cb/a1b/F1b/c1b, at/ct/a1t/F1t/c1t)
        into this model. Only this model's new `condition_net.*` keys are
        allowed to be missing; any OTHER missing/unexpected key is an error
        (likely a real shape/name mismatch, not just the added conditioning)."""
        sd = torch.load(path, map_location=map_location, weights_only=False)
        if 'model_state_dict' in sd:
            sd = sd['model_state_dict']
        missing, unexpected = self.load_state_dict(sd, strict=False)
        bad_missing = [k for k in missing if not k.startswith('condition_net.')]
        if bad_missing or unexpected:
            raise RuntimeError(f"Unexpected state_dict mismatch beyond condition_net.*: "
                               f"missing={bad_missing}, unexpected={unexpected}")
        return missing  # the condition_net.* keys that stayed at their zero-init
