"""
Branch-Conditioned Gabor Fusion_DeepONet: extends model_fusion_garbor_deeponet.py
(not modified by this file) by making the TrunkNet's Gabor activation
parameters (w, s, phi) depend on the branch input u=(Ma, Kn):

    g_l(z; u) = exp(-s_l(u)^2 * z^2) * cos(w_l(u) * z + phi_l(u))

    w_l(u)   = w_l^base   + epsilon * tanh(dw_l(u))
    s_l(u)   = s_l^base   + epsilon * tanh(ds_l(u))
    phi_l(u) = phi_l^base + epsilon * tanh(dphi_l(u))

w_l^base/s_l^base/phi_l^base are learnable nn.Parameters (init 1.0/1.0/0.0,
same as the unconditioned model). (dw_l, ds_l, dphi_l) for all layers come
from a single small Conditioning MLP applied once to the ORIGINAL branch
input (before the BranchNet loop below overwrites x_branch):
Linear(branch_dim, 32) -> GELU -> Linear(32, depth*3). Its last Linear is
zero-initialized, so at init every correction is exactly 0 and this model
is numerically identical to the unconditioned Gabor Fusion_DeepONet.

Only the TrunkNet's Gabor activations are conditioned. The BranchNet's
Gabor activations stay the original static per-layer GaborActivation
(imported from model_fusion_garbor_deeponet.py, unchanged). The fusion
einsum, skip accumulation, and final DeepONet inner product are unchanged.

Parameter names (`branch_net`, `trunk_net`, `branch_acts`, and `trunk_acts[i].w`/
`.s`/`.phi`) are kept identical to model_fusion_garbor_deeponet.py so a
state_dict from that model loads into this one with `strict=False` (only
the new `cond_mlp.*` and this model's extra per-layer buffers are missing).
"""
import torch
import torch.nn as nn

from model_fusion_garbor_deeponet import GaborActivation


class ConditionalGaborActivation(nn.Module):
    """Gabor activation whose (w, s, phi) are a learnable per-layer base
    plus a bounded, branch-conditioned correction (see module docstring)."""

    def __init__(self, w0=1.0, s0=1.0, phi0=0.1, epsilon=0.1):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(float(w0)))
        self.s = nn.Parameter(torch.tensor(float(s0)))
        self.phi = nn.Parameter(torch.tensor(float(phi0)))
        self.epsilon = epsilon

    def forward(self, z, delta):
        """
        Args:
            z: [B, N, hidden_dim] -- pre-activation trunk features.
            delta: [B, 3] -- this layer's (dw, ds, dphi) conditioning MLP
                output, one scalar triple per batch sample.
        """
        dw = delta[:, 0].view(-1, 1, 1)
        ds = delta[:, 1].view(-1, 1, 1)
        dphi = delta[:, 2].view(-1, 1, 1)

        w_l = self.w + self.epsilon * torch.tanh(dw)
        s_l = self.s + self.epsilon * torch.tanh(ds)
        phi_l = self.phi + self.epsilon * torch.tanh(dphi)

        envelope = torch.exp(-(s_l ** 2) * z ** 2)
        return envelope * torch.cos(w_l * z + phi_l)


class Fusion_DeepONet(nn.Module):
    def __init__(self, branch_dim=1, trunk_dim=2, hidden_dim=280, num_outputs=10, depth=5,
                 activation='GELU', epsilon=0.1, cond_hidden_dim=32):
        """
        Same constructor signature as model_fusion_garbor_deeponet.Fusion_DeepONet,
        plus two new optional kwargs:
            epsilon: bound on the branch-conditioned correction to each Gabor
                parameter (default 0.1, configurable).
            cond_hidden_dim: hidden width of the Conditioning MLP (default 32,
                per spec: Linear(branch_dim, 32) -> GELU -> Linear(32, depth*3)).
        """
        super().__init__()

        self.depth = depth
        self.epsilon = epsilon

        # BranchNet Gabor activations: unchanged, static per-layer (w, s),
        # imported as-is from model_fusion_garbor_deeponet.py.
        self.branch_acts = nn.ModuleList([GaborActivation(w0=1.0, s0=1.0) for _ in range(depth)])
        # TrunkNet Gabor activations: branch-conditioned (this model's only change).
        self.trunk_acts = nn.ModuleList([ConditionalGaborActivation(w0=1.0, s0=1.0, phi0=0.1, epsilon=epsilon)
                                          for _ in range(depth)])

        # Conditioning MLP: x_branch=(Ma, Kn) -> per-layer (dw, ds, dphi) for
        # the TrunkNet's Gabor activations. Last Linear zero-initialized so
        # the model starts identical to the unconditioned Gabor model.
        self.cond_mlp = nn.Sequential(
            nn.Linear(branch_dim, cond_hidden_dim),
            nn.GELU(),
            nn.Linear(cond_hidden_dim, depth * 3),
        )
        nn.init.zeros_(self.cond_mlp[-1].weight)
        nn.init.zeros_(self.cond_mlp[-1].bias)

        # Fusion-deeponet bookkeeping (unchanged): self.L-1 == depth hidden
        # layers with an activation, plus one final (un-activated) output layer.
        self.L = depth + 1

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
        # overwrites x_branch -- used only to condition the TrunkNet's Gabor
        # activations.
        x_branch_orig = x_branch

        cond = self.cond_mlp(x_branch_orig)         # [B, depth*3]
        cond = cond.view(-1, self.depth, 3)          # [B, depth, 3]

        skip = []

        for i in range(self.L - 1):

            branch_lin = self.branch_net[i](x_branch)
            x_branch = self.branch_acts[i](branch_lin)
            skip.append(x_branch)

        for i in range(1, self.L - 1):
            skip[i] = skip[i - 1] + skip[i]

        for i in range(self.L - 1):

            trunk_lin = self.trunk_net[i](x_trunk)
            x_trunk = self.trunk_acts[i](trunk_lin, cond[:, i, :])

            x_trunk = torch.einsum('bk,bik->bik', skip[i], x_trunk)

        x_branch = self.branch_net[-1](x_branch)
        x_trunk = self.trunk_net[-1](x_trunk)

        B_out_reshaped = x_branch.view(-1, self.num_outputs, self.hidden_dim)
        # [Batch, num_outputs, hidden_dim]

        prediction = torch.einsum("bnk, bik -> bin", B_out_reshaped, x_trunk)
        # [Batch, N_points, num_outputs]

        return prediction
