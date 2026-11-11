"""
Model A -- HyperDeepONet-Gabor: ablation of model_hyperdeeponet.py (not
modified by this file) that replaces the TrunkNet's GELU/Tanh activation
with a real-valued Gabor activation:

    g_l(z) = exp(-s_l^2 * z^2) * cos(w_l * z + phi_l)

One independent (w_l, s_l, phi_l) triple of learnable scalars per trunk
hidden layer (trunk_depth of them), shared by every neuron and every sample
within that layer -- i.e. NOT branch-conditioned (see
model_hyperdeeponet_gabor_cond.py, Model B, for the branch-conditioned
version). Init: w=1.0, s=1.0, phi=0.0. The final trunk output Linear (the
(trunk_depth+1)-th branch-generated layer) still has no activation, exactly
as in the original.

The BranchNet (a plain _MLP with GELU/Tanh, generating ALL of the TrunkNet's
weights/biases) is unchanged. Constructor signature, trunk weight/bias
generation + slicing, batched einsum, and output shape all match
model_hyperdeeponet.py; branch_dim is a free constructor arg (674 for
Task1_Airfoil, 2 for Task2_Cylinder -- nothing here is hardcoded to either).
"""

import torch
import torch.nn as nn


class GaborActivation(nn.Module):
    """Real-valued Gabor activation: g(z) = exp(-s^2 z^2) cos(w z + phi).
    w, s, phi are learnable, layerwise SCALAR parameters (shared by every
    neuron and every sample in the layer)."""

    def __init__(self, w0=1.0, s0=1.0, phi0=0.1):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(float(w0)))
        self.s = nn.Parameter(torch.tensor(float(s0)))
        self.phi = nn.Parameter(torch.tensor(float(phi0)))

    def forward(self, z):
        envelope = torch.exp(-(self.s ** 2) * z ** 2)
        return envelope * torch.cos(self.w * z + self.phi)


class HyperDeepONet(nn.Module):
    def __init__(self, branch_dim=674, trunk_dim=2, hidden_dim=46,
                 num_outputs=4, trunk_depth=3, branch_depth=3,
                 activation='GELU'):
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

        # One independent Gabor activation per trunk hidden layer.
        self.trunk_acts = nn.ModuleList([GaborActivation(w0=1.0, s0=1.0, phi0=0.1)
                                          for _ in range(trunk_depth)])

        # Total parameters needed to construct the trunk net (unchanged).
        t_para = 0
        for i in range(len(self.trunk_dims) - 1):
            t_para += self.trunk_dims[i] * self.trunk_dims[i + 1] + self.trunk_dims[i + 1]

        # Branch: single network -> t_para (trunk weights/biases). Unchanged
        # (still uses GELU/Tanh, not Gabor).
        branch_dims = [branch_dim] + [hidden_dim] * branch_depth + [t_para]
        self.branch_net = _MLP(branch_dims, act)

        self.num_outputs = num_outputs

    def _branch_forward(self, x):
        """Run branch net on x -> trunk parameters."""
        return self.branch_net(x)  # [B, t_para]

    def _trunk_forward(self, params, x_trunk):
        """Hypernetwork trunk: params -> weights/biases -> forward pass."""
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
            y = self.trunk_acts[i](y)

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
        params = self._branch_forward(x_branch)  # [B, t_para]
        return self._trunk_forward(params, x_trunk)


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
