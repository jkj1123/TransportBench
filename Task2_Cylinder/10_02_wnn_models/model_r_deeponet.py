"""
R-DeepONet: standard DeepONet (Lu et al.) with the trunk net replaced by a
RIDGELET basis layer, instead of an MLP or the isotropic/anisotropic
Mexican-hat WNN used in model_w_deeponet.py.

Branch net: unchanged from BoltzmannDeepONet -- plain MLP, outputs
num_basis * num_outputs branch coefficients.

Trunk net: a single RidgeletTrunk layer. Each of `num_basis` units computes
a 1D Mexican-hat wavelet of the coordinate's projection onto a learned
DIRECTION theta_i, not a localized blob around a center:

    psi_i(x) = (1 - u_i^2) * exp(-u_i^2 / 2),
    u_i = (<theta_i, x> - b_i) / a_i,
    theta_i = (cos(phi_i), sin(phi_i))   (unit vector, trunk_dim=2 assumed)

Unlike WaveletTrunk's basis (localized, decays in every direction away from
a center -> an axis-aligned ellipse footprint), a ridgelet is CONSTANT in
every direction orthogonal to theta_i -- it only varies (oscillates,
localizes) along that one direction, giving a "ridge" / plane-wave-like
footprint instead of a blob. This is the natural multi-D generalization of
a 1D wavelet for representing line/hyperplane-aligned structure (Candes,
1998) -- e.g. a wake elongated downstream, a boundary layer hugging a
surface, a shock front -- which a sum of localized blobs represents only
inefficiently. Note a plain MLP neuron act(w.x+b) is ALSO a ridge function
(constant orthogonal to w); a ridgelet is the same idea with an oscillating
wavelet in place of a monotonic activation.

The branch/trunk outputs are combined exactly as in the original DeepONet
(dot product over the shared basis dimension), so this is a drop-in
trunk-only ablation of BoltzmannDeepONet / W_DeepONet.
"""

import math
import torch
import torch.nn as nn


class RidgeletTrunk(nn.Module):
    """Ridgelet basis layer: R^2 -> R^num_basis (trunk_dim must be 2 -- the
    direction is parameterized by a single learnable angle, which keeps it
    unit-norm automatically via cos/sin, no explicit normalization needed)."""

    def __init__(self, trunk_dim, num_basis, coord_range=None):
        super().__init__()
        assert trunk_dim == 2, "RidgeletTrunk's angle parameterization assumes trunk_dim=2"
        self.trunk_dim = trunk_dim
        self.num_basis = num_basis

        if coord_range is None:
            lo = torch.tensor([-1.0, -1.0])
            hi = torch.tensor([1.0, 1.0])
        else:
            lo = torch.tensor([c[0] for c in coord_range], dtype=torch.float32)
            hi = torch.tensor([c[1] for c in coord_range], dtype=torch.float32)

        # Direction: learnable angle phi_i, random init over [0, 2*pi).
        # theta_i = (cos(phi_i), sin(phi_i)) is always unit-norm by
        # construction -- no projection/renormalization step needed.
        phi_init = torch.rand(num_basis) * 2 * math.pi
        self.phi = nn.Parameter(phi_init)

        # Init b_i, a_i from the ACTUAL projection range of the coordinate
        # domain's 4 corners onto each basis's own initial direction (so
        # translations/scales start out matched to the real physical grid,
        # same spirit as WaveletTrunk's coord_range-based init).
        corners = torch.stack([
            torch.stack([lo[0], lo[1]]), torch.stack([lo[0], hi[1]]),
            torch.stack([hi[0], lo[1]]), torch.stack([hi[0], hi[1]]),
        ])  # [4, 2]
        theta_init = torch.stack([torch.cos(phi_init), torch.sin(phi_init)], dim=-1)  # [P, 2]
        proj_corners = corners @ theta_init.t()  # [4, P]
        proj_min = proj_corners.min(dim=0).values  # [P]
        proj_max = proj_corners.max(dim=0).values  # [P]

        b_init = proj_min + torch.rand(num_basis) * (proj_max - proj_min)
        self.b = nn.Parameter(b_init)

        span = (proj_max - proj_min).clamp(min=1e-3)
        a_init = (span / num_basis).clamp(min=1e-3)  # rough 1D tiling spacing along each own direction
        self.log_a = nn.Parameter(torch.log(a_init))

    def forward(self, x_trunk):
        """
        Args:
            x_trunk: [N, trunk_dim] or [B, N, trunk_dim]
        Returns:
            [N, num_basis] or [B, N, num_basis]
        """
        theta = torch.stack([torch.cos(self.phi), torch.sin(self.phi)], dim=-1)  # [P, 2]
        proj = torch.einsum('...d,pd->...p', x_trunk, theta)  # [..., N, P]
        a = torch.exp(self.log_a)  # [P], kept positive
        u = (proj - self.b) / a
        psi = (1.0 - u ** 2) * torch.exp(-0.5 * u ** 2)  # 1D Mexican-hat along theta_i
        return psi


class R_DeepONet(nn.Module):
    def __init__(self, branch_dim, trunk_dim, hidden_dim=256, num_basis=124,
                 num_outputs=4, depth=5, activation='GELU', coord_range=None):
        """DeepONet with a Ridgelet trunk (see module docstring)."""
        super().__init__()

        if activation == 'GELU':
            self.act = nn.GELU()
        elif activation == 'Tanh':
            self.act = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        branch_layers = []
        branch_layers.extend([nn.Linear(branch_dim, hidden_dim), self.act])
        for _ in range(depth - 1):
            branch_layers.extend([nn.Linear(hidden_dim, hidden_dim), self.act])
        branch_layers.append(nn.Linear(hidden_dim, num_basis * num_outputs))
        self.branch_net = nn.Sequential(*branch_layers)

        self.trunk_net = RidgeletTrunk(trunk_dim, num_basis, coord_range=coord_range)

        self.num_outputs = num_outputs
        self.num_basis = num_basis

    def forward(self, x_branch, x_trunk):
        B_out = self.branch_net(x_branch)  # [Batch, num_basis * num_outputs]
        T_out = self.trunk_net(x_trunk)     # [N_points, num_basis]

        B_out_reshaped = B_out.view(-1, self.num_outputs, self.num_basis)
        prediction = torch.einsum("bkh, nh -> bkn", B_out_reshaped, T_out)
        prediction = prediction.permute(0, 2, 1)
        return prediction
