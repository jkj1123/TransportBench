"""
Model B -- HyperDeepONet-ConditionalGabor: same Gabor activation as Model A
(model_hyperdeeponet_gabor.py), but the TrunkNet's Gabor parameters depend
on the branch input u = x_branch:

    g_l(z; u) = exp(-s_l(u)^2 * z^2) * cos(w_l(u) * z + phi_l(u))

    w_l(u)   = w_l^base   + epsilon * tanh(dw_l(u))
    s_l(u)   = s_l^base   + epsilon * tanh(ds_l(u))
    phi_l(u) = phi_l^base + epsilon * tanh(dphi_l(u))

w_l^base/s_l^base/phi_l^base are learnable nn.Parameters, initialized
exactly as in Model A (1.0/1.0/0.0). All (dw_l, ds_l, dphi_l) for every
trunk hidden layer come from a single Conditioning MLP applied once to the
raw branch input u (not updated/reused anywhere else in this model, unlike
Fusion_DeepONet's branch tower, so there is no "original vs. updated
x_branch" ambiguity here):

    self.condition_net = nn.Sequential(
        nn.Linear(branch_dim, 32), nn.GELU(), nn.Linear(32, trunk_depth * 3))

Its last Linear is zero-initialized, so at init every correction is exactly
0 and this model is numerically identical to Model A. BatchNorm is not
used anywhere in condition_net.

Everything else (BranchNet generating ALL of the TrunkNet's weights/biases,
weight/bias slicing, batched einsum, final no-activation output layer,
output shape) is unchanged from model_hyperdeeponet.py / Model A, and the
code structure deliberately mirrors Model A as closely as possible for a
fair, apples-to-apples comparison. branch_dim is a free constructor arg
(674 for Task1_Airfoil, 2 for Task2_Cylinder).
"""

import torch
import torch.nn as nn


class ConditionalGaborActivation(nn.Module):
    """Gabor activation whose (w, s, phi) are a learnable per-layer base plus
    a bounded, branch-conditioned correction (see module docstring)."""

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
            delta: [B, 3] -- this layer's (dw, ds, dphi), one scalar triple
                per batch sample (shared across N and hidden_dim).
        """
        dw = delta[:, 0].view(-1, 1, 1)
        ds = delta[:, 1].view(-1, 1, 1)
        dphi = delta[:, 2].view(-1, 1, 1)

        w_l = self.w + self.epsilon * torch.tanh(dw)
        s_l = self.s + self.epsilon * torch.tanh(ds)
        phi_l = self.phi + self.epsilon * torch.tanh(dphi)

        envelope = torch.exp(-(s_l ** 2) * z ** 2)
        return envelope * torch.cos(w_l * z + phi_l)


class HyperDeepONet(nn.Module):
    def __init__(self, branch_dim=674, trunk_dim=2, hidden_dim=46,
                 num_outputs=4, trunk_depth=3, branch_depth=3,
                 activation='GELU', epsilon=0.1, cond_hidden_dim=32):
        """
        Same constructor signature as model_hyperdeeponet.HyperDeepONet /
        Model A, plus two new optional kwargs:
            epsilon: bound on the branch-conditioned correction to each
                Gabor parameter (default 0.1, configurable).
            cond_hidden_dim: hidden width of condition_net (default 32, per
                spec: Linear(branch_dim, 32) -> GELU -> Linear(32, trunk_depth*3)).
        """
        super().__init__()

        if activation == 'Tanh':
            act = nn.Tanh
        elif activation == 'GELU':
            act = nn.GELU
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Trunk architecture: [trunk_dim, hidden, ..., hidden, num_outputs]
        self.trunk_dims = [trunk_dim] + [hidden_dim] * trunk_depth + [num_outputs]
        self.trunk_depth = trunk_depth
        self.epsilon = epsilon

        # One independent, branch-conditioned Gabor activation per trunk
        # hidden layer; base params initialized exactly as in Model A.
        self.trunk_acts = nn.ModuleList([ConditionalGaborActivation(w0=1.0, s0=1.0, phi0=0.1, epsilon=epsilon)
                                          for _ in range(trunk_depth)])

        # Conditioning MLP: u=x_branch -> per-layer (dw, ds, dphi) for the
        # TrunkNet's Gabor activations. Last Linear zero-initialized so the
        # model starts identical to Model A.
        self.condition_net = nn.Sequential(
            nn.Linear(branch_dim, cond_hidden_dim),
            nn.GELU(),
            nn.Linear(cond_hidden_dim, trunk_depth * 3),
        )
        nn.init.zeros_(self.condition_net[-1].weight)
        nn.init.zeros_(self.condition_net[-1].bias)

        # Total parameters needed to construct the trunk net (unchanged).
        t_para = 0
        for i in range(len(self.trunk_dims) - 1):
            t_para += self.trunk_dims[i] * self.trunk_dims[i + 1] + self.trunk_dims[i + 1]

        # Branch: single network -> t_para (trunk weights/biases). Unchanged
        # (still uses GELU/Tanh, not Gabor) -- separate from condition_net.
        branch_dims = [branch_dim] + [hidden_dim] * branch_depth + [t_para]
        self.branch_net = _MLP(branch_dims, act)

        self.num_outputs = num_outputs

    def _branch_forward(self, x):
        """Run branch net on x -> trunk parameters."""
        return self.branch_net(x)  # [B, t_para]

    def _trunk_forward(self, params, x_trunk, cond):
        """Hypernetwork trunk: params -> weights/biases -> forward pass.

        cond: [B, trunk_depth, 3] -- per-layer, per-sample (dw, ds, dphi).
        """
        B = params.shape[0]  # use branch batch size (handles shared trunk)
        # Normalize to 3D: [B, N, trunk_dim]
        if x_trunk.dim() == 2:
            x_trunk = x_trunk.unsqueeze(0).expand(B, -1, -1)

        _, N, _ = x_trunk.shape
        y = x_trunk  # [B, N, trunk_dim]
        start = 0

        for i in range(len(self.trunk_dims) - 2):
            d_in, d_out = self.trunk_dims[i], self.trunk_dims[i + 1]

            w_sz = d_in * d_out
            weight = params[:, start:start + w_sz].reshape(B, d_out, d_in)
            start += w_sz
            bias = params[:, start:start + d_out].reshape(B, 1, d_out)
            start += d_out

            y = torch.einsum("bij,bgj->bgi", weight, y) + bias  # [B, N, d_out]
            y = self.trunk_acts[i](y, cond[:, i, :])

        # Last layer: no activation
        d_in, d_out = self.trunk_dims[-2], self.trunk_dims[-1]
        w_sz = d_in * d_out
        weight = params[:, start:start + w_sz].reshape(B, d_out, d_in)
        start += w_sz
        bias = params[:, start:start + d_out].reshape(B, 1, d_out)

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
        params = self._branch_forward(x_branch)             # [B, t_para]
        cond = self.condition_net(x_branch)                   # [B, trunk_depth*3]
        cond = cond.view(-1, self.trunk_depth, 3)             # [B, trunk_depth, 3]
        return self._trunk_forward(params, x_trunk, cond)


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
