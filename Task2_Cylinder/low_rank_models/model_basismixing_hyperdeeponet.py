"""
Basis-Mixing HyperDeepONet: ablation of model_hyperdeeponet.py (not modified
by this file) that splits the single hypernetwork branch into two separate
networks:

  Branch 1 (weight/bias BASIS generator): a single shared MLP that takes a
  SCALAR, sample-INDEPENDENT mode index k (k=1..num_basis, normalized) and
  outputs that mode's full flattened (weight, bias) parameter vector for
  EVERY trunk layer (including the final output layer). Called once per
  forward pass with all num_basis indices stacked as a batch of size
  num_basis -> output [num_basis, t_para]. This does not depend on the
  branch input (Ma, Kn) at all, and the same num_basis bases are shared by
  every sample in the batch.

  Branch 2 (per-sample MIXING COEFFICIENT generator): a small MLP that
  takes u=(Ma, Kn) ([B, branch_dim]) and outputs [B, L, num_basis] -- one
  length-num_basis coefficient vector per trunk layer, per sample (L = the
  total number of trunk Linear layers, INCLUDING the final output layer).

  The two are combined per trunk layer l as

      W_l(u) = (1/sqrt(num_basis)) * sum_k s_{l,k}(u) * W_{l,k}
      b_l(u) = (1/sqrt(num_basis)) * sum_k s_{l,k}(u) * b_{l,k}

  (weight and bias for a given layer/mode always share the same mixing
  coefficient s_{l,k}(u), since they come from the same basis-k block of
  Branch 1's output and are mixed with the same per-layer coefficient from
  Branch 2). No base/residual weight path (no W_0) is added on top.

  Once the per-sample, per-layer (weight, bias) are mixed, the batched
  trunk forward pass is byte-for-byte the SAME computation as
  model_hyperdeeponet.HyperDeepONet._trunk_forward (same slicing order,
  same per-layer activation, same un-activated final layer, same output
  shape) -- the only thing that changes is how `params` ([B, t_para]) is
  produced.

Trunk architecture (trunk_dims, t_para layout, GELU/Tanh support, 2D/3D
x_trunk handling, [B, N, num_outputs] output) and the overall
branch/trunk-coordinate interface are unchanged from the original.
Branch 1 and Branch 2 each have their own independently configurable
hidden_dim/depth.
"""

import math
import torch
import torch.nn as nn


class BasisMixingHyperDeepONet(nn.Module):
    def __init__(self, branch_dim=674, trunk_dim=2, hidden_dim=46,
                 num_outputs=4, trunk_depth=3,
                 num_basis=20,
                 branch1_hidden_dim=32, branch1_depth=2,
                 branch2_hidden_dim=32, branch2_depth=2,
                 activation='GELU', coeff_offset=0.0):
        """
        coeff_offset: fixed constant added to every element of Branch 2's
        raw mixing-coefficient output (every layer, every mode, every
        sample) before mixing. coeff_offset=0.0 (default) reproduces the
        original behavior exactly. Observed failure mode without it: one
        trunk layer's mixing coefficients collapse to exactly 0 for every
        sample (confirmed via diagnose()/dead_layer_warning on a trained
        checkpoint), zeroing that layer's weight+bias and permanently
        severing the trunk's dependence on the spatial query coordinate
        from that layer on (no W_0/residual path exists to fall back on,
        per spec). A nonzero offset keeps all 20 modes from simultaneously
        landing on exactly -coeff_offset, which is a much less likely fixed
        point than all 20 independently landing on exactly 0.
        """
        super().__init__()
        self.coeff_offset = coeff_offset

        if activation == 'Tanh':
            act = nn.Tanh
            self._trunk_act = torch.tanh
        elif activation == 'GELU':
            act = nn.GELU
            self._trunk_act = nn.functional.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Trunk architecture: [trunk_dim, hidden, ..., hidden, num_outputs]
        # (same as the original -- the trunk itself has no parameters of its
        # own; this just defines the shapes to generate/mix).
        self.trunk_dims = [trunk_dim] + [hidden_dim] * trunk_depth + [num_outputs]
        self.num_basis = num_basis

        # Per-layer (weight_size + bias_size) block sizes, in the SAME order
        # the original model concatenates them into `t_para` -- L = number
        # of trunk Linear layers, including the final (unactivated) output
        # layer.
        self.block_sizes = [self.trunk_dims[i] * self.trunk_dims[i + 1] + self.trunk_dims[i + 1]
                             for i in range(len(self.trunk_dims) - 1)]
        self.L = len(self.block_sizes)
        t_para = sum(self.block_sizes)
        self.t_para = t_para

        # --- Branch 1: weight/bias BASIS generator ---
        # Input: a single scalar (normalized mode index). Shared MLP, called
        # once per forward with all num_basis indices stacked as a batch.
        # Fixed (non-learnable) normalized indices in [-1, 1].
        mode_idx = torch.linspace(-1.0, 1.0, num_basis).view(num_basis, 1)
        self.register_buffer('mode_idx', mode_idx)
        branch1_dims = [1] + [branch1_hidden_dim] * branch1_depth + [t_para]
        self.branch1_net = _MLP(branch1_dims, act)

        # --- Branch 2: per-sample MIXING COEFFICIENT generator ---
        # Input: u=(Ma, Kn), i.e. the ordinary branch_dim-dim branch input.
        # Output: [B, L * num_basis], reshaped to [B, L, num_basis].
        branch2_dims = [branch_dim] + [branch2_hidden_dim] * branch2_depth + [self.L * num_basis]
        self.branch2_net = _MLP(branch2_dims, act)

        self.num_outputs = num_outputs

    def _basis_forward(self):
        """Branch 1: shared, sample-independent weight/bias bases.
        Returns: [num_basis, t_para]."""
        return self.branch1_net(self.mode_idx)

    def _mixing_forward(self, x_branch):
        """Branch 2: per-sample, per-layer mixing coefficients.
        Returns: [B, L, num_basis]."""
        B = x_branch.shape[0]
        return self.branch2_net(x_branch).view(B, self.L, self.num_basis) + self.coeff_offset

    def _mix_params(self, basis, coeffs):
        """Combine Branch 1's bases and Branch 2's mixing coefficients into
        the same flattened [B, t_para] layout model_hyperdeeponet.py's
        _trunk_forward expects (per-layer weight-then-bias blocks, in order).

        Args:
            basis:  [num_basis, t_para]
            coeffs: [B, L, num_basis]

        Returns:
            [B, t_para]
        """
        scale = 1.0 / math.sqrt(self.num_basis)
        mixed_blocks = []
        start = 0
        for l in range(self.L):
            block_l = self.block_sizes[l]
            basis_l = basis[:, start:start + block_l]          # [num_basis, block_l]
            coeffs_l = coeffs[:, l, :]                          # [B, num_basis]
            mixed_l = scale * torch.einsum('bk,kp->bp', coeffs_l, basis_l)  # [B, block_l]
            mixed_blocks.append(mixed_l)
            start += block_l
        return torch.cat(mixed_blocks, dim=1)  # [B, t_para]

    def _trunk_forward(self, params, x_trunk):
        """Hypernetwork trunk: params -> weights/biases -> forward pass.
        Identical to model_hyperdeeponet.HyperDeepONet._trunk_forward."""
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
            y = self._trunk_act(y)

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
            x_branch: [B, branch_dim]  sensor values (Ma, Kn)
            x_trunk:  [N, trunk_dim] or [B, N, trunk_dim]  query coordinates

        Returns:
            [B, N, num_outputs]
        """
        basis = self._basis_forward()            # [num_basis, t_para]
        coeffs = self._mixing_forward(x_branch)   # [B, L, num_basis]
        params = self._mix_params(basis, coeffs)  # [B, t_para]
        return self._trunk_forward(params, x_trunk)

    # ---- diagnostics ----

    def diagnose(self, x_branch, x_trunk, verbose=True):
        """Returns (and optionally prints) basis/mixing/param/activation
        scale statistics, and flags likely dead-trunk collapse (mixing
        coefficients ~0 for some/all layers)."""
        with torch.no_grad():
            basis = self._basis_forward()
            coeffs = self._mixing_forward(x_branch)
            params = self._mix_params(basis, coeffs)

            stats = {
                'basis_std': basis.std().item(),
                'basis_absmax': basis.abs().max().item(),
                'coeffs_std': coeffs.std().item(),
                'coeffs_absmax': coeffs.abs().max().item(),
                'params_std': params.std().item(),
                'params_absmax': params.abs().max().item(),
                'has_nan': bool(torch.isnan(params).any() or torch.isnan(basis).any() or torch.isnan(coeffs).any()),
                'has_inf': bool(torch.isinf(params).any() or torch.isinf(basis).any() or torch.isinf(coeffs).any()),
            }
            # Per-layer mixing-coefficient RMS, to catch a specific layer
            # collapsing toward 0 (dead trunk layer) even if the overall
            # average looks fine.
            per_layer_rms = coeffs.pow(2).mean(dim=(0, 2)).sqrt()  # [L]
            stats['per_layer_coeff_rms'] = per_layer_rms.tolist()
            stats['dead_layer_warning'] = [l for l, v in enumerate(per_layer_rms.tolist()) if v < 1e-3]

            y = self._trunk_forward(params, x_trunk)
            stats['output_std'] = y.std().item()
            stats['output_absmax'] = y.abs().max().item()
            stats['output_has_nan'] = bool(torch.isnan(y).any())
            stats['output_has_inf'] = bool(torch.isinf(y).any())

        if verbose:
            print("=== BasisMixingHyperDeepONet diagnostics ===")
            for k, v in stats.items():
                print(f"  {k}: {v}")
        return stats


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


def count_params(branch_dim=2, trunk_dim=2, hidden_dim=46, num_outputs=4, trunk_depth=3,
                  num_basis=20, branch1_hidden_dim=32, branch1_depth=2,
                  branch2_hidden_dim=32, branch2_depth=2, activation='GELU'):
    """Instantiate a BasisMixingHyperDeepONet with the given config and
    return (total_params, breakdown_dict), without needing to build a full
    training run -- used for the ~20K parameter-budget search."""
    m = BasisMixingHyperDeepONet(branch_dim=branch_dim, trunk_dim=trunk_dim, hidden_dim=hidden_dim,
                                 num_outputs=num_outputs, trunk_depth=trunk_depth, num_basis=num_basis,
                                 branch1_hidden_dim=branch1_hidden_dim, branch1_depth=branch1_depth,
                                 branch2_hidden_dim=branch2_hidden_dim, branch2_depth=branch2_depth,
                                 activation=activation)
    b1 = sum(p.numel() for p in m.branch1_net.parameters())
    b2 = sum(p.numel() for p in m.branch2_net.parameters())
    total = b1 + b2
    return total, {'branch1_params': b1, 'branch2_params': b2, 't_para': m.t_para, 'L': m.L}
