"""
WHM-DeepONet: same two-sub-trunk idea as WM-DeepONet (model_wm_deeponet.py)
-- a WNN trunk + an MLP trunk, combined via learnable scalars a, b -- but
the MLP sub-trunk is now HYPERNETWORK-GENERATED instead of directly
learned: the branch net outputs that trunk MLP's full weights/biases for
each sample, exactly like HyperDeepONet's trunk (model_hyperdeeponet.py),
rather than the trunk having its own fixed nn.Linear parameters.

    trunk(x) = a * WaveletTrunk(x) + b * HyperMLPTrunk(x; params(u))

where `params(u)` is produced by the branch net from the branch input u
(a second output head, alongside the usual num_basis*num_outputs branch
coefficients used in the dot-product combine).

WaveletTrunk (anisotropic Mexican-hat WNN) is unchanged, imported from
model_w_deeponet.py. The hyper-generated trunk MLP uses the same
[trunk_dim, hidden, hidden, hidden, hidden, hidden, num_basis] layer shape
as WM-DeepONet's directly-learned MLPTrunk (depth=5), so this is a clean
"directly-learned vs hypernetwork-generated" ablation of that one sub-trunk.
"""

import torch
import torch.nn as nn

from model_w_deeponet import WaveletTrunk


class HyperMLPTrunk(nn.Module):
    """Trunk MLP whose weights/biases are supplied externally (per-sample),
    not stored as nn.Parameters. Architecture: [trunk_dim, hidden_dim]*depth
    -> num_basis (matches MLPTrunk's depth=5 layer pattern)."""

    def __init__(self, trunk_dim, hidden_dim, num_basis, depth, activation):
        super().__init__()
        self.trunk_dims = [trunk_dim] + [hidden_dim] * depth + [num_basis]
        self._act = torch.tanh if activation == 'Tanh' else torch.nn.functional.gelu

        self.param_size = 0
        for i in range(len(self.trunk_dims) - 1):
            self.param_size += self.trunk_dims[i] * self.trunk_dims[i + 1] + self.trunk_dims[i + 1]

    def forward(self, params, x_trunk):
        """
        Args:
            params: [B, param_size] -- this sample's trunk weights/biases
            x_trunk: [N, trunk_dim] (shared across the batch)
        Returns:
            [B, N, num_basis]
        """
        B = params.shape[0]
        y = x_trunk.unsqueeze(0).expand(B, -1, -1)  # [B, N, trunk_dim]
        start = 0
        n_layers = len(self.trunk_dims) - 1
        for i in range(n_layers):
            d_in, d_out = self.trunk_dims[i], self.trunk_dims[i + 1]
            w_sz = d_in * d_out
            weight = params[:, start:start + w_sz].reshape(B, d_out, d_in)
            start += w_sz
            bias = params[:, start:start + d_out].reshape(B, 1, d_out)
            start += d_out
            y = torch.einsum("bij,bgj->bgi", weight, y) + bias
            if i < n_layers - 1:
                y = self._act(y)
        return y  # [B, N, num_basis] -- last layer has no activation


class WHM_DeepONet(nn.Module):
    def __init__(self, branch_dim, trunk_dim, hidden_dim=256, trunk_hidden_dim=None,
                 num_basis=124, num_outputs=4, depth=5, activation='GELU', coord_range=None):
        """
        `hidden_dim`: branch net's shared backbone width.
        `trunk_hidden_dim`: hidden width of the hypernetwork-GENERATED MLP
            trunk (kept small -- generating its weights is what's expensive).
            Defaults to `hidden_dim` if not given.
        `num_basis`: shared WNN / hyper-trunk output width (p in the
            DeepONet dot product).
        """
        super().__init__()
        if trunk_hidden_dim is None:
            trunk_hidden_dim = hidden_dim

        if activation == 'GELU':
            self.act = nn.GELU()
        elif activation == 'Tanh':
            self.act = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Hyper-generated trunk: determines how many params the branch's
        # second head must output.
        self.hyper_trunk = HyperMLPTrunk(trunk_dim, trunk_hidden_dim, num_basis, depth, activation)
        t_para = self.hyper_trunk.param_size

        # Shared branch backbone -> two heads: (1) dot-product coefficients
        # [num_basis * num_outputs], (2) hyper-trunk weights [t_para].
        backbone = []
        backbone.extend([nn.Linear(branch_dim, hidden_dim), self.act])
        for _ in range(depth - 1):
            backbone.extend([nn.Linear(hidden_dim, hidden_dim), self.act])
        self.branch_backbone = nn.Sequential(*backbone)
        self.head_coeffs = nn.Linear(hidden_dim, num_basis * num_outputs)
        self.head_trunk_params = nn.Linear(hidden_dim, t_para)

        # Directly-learned WNN sub-trunk (unchanged from W/WM-DeepONet).
        self.trunk_wavelet = WaveletTrunk(trunk_dim, num_basis, coord_range=coord_range)

        # Learnable combination weights (init 1.0 = plain sum at start).
        self.a = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(1.0))

        self.num_outputs = num_outputs
        self.num_basis = num_basis

    def forward(self, x_branch, x_trunk):
        feat = self.branch_backbone(x_branch)         # [B, hidden_dim]
        B_out = self.head_coeffs(feat)                 # [B, num_basis * num_outputs]
        trunk_params = self.head_trunk_params(feat)     # [B, t_para]

        T_wavelet = self.trunk_wavelet(x_trunk)          # [N, num_basis] (shared across batch)
        T_hyper = self.hyper_trunk(trunk_params, x_trunk)  # [B, N, num_basis]

        T_out = self.a * T_wavelet.unsqueeze(0) + self.b * T_hyper  # [B, N, num_basis]

        B_out_reshaped = B_out.view(-1, self.num_outputs, self.num_basis)  # [B, K, num_basis]

        prediction = torch.einsum("bkh, bnh -> bkn", B_out_reshaped, T_out)
        prediction = prediction.permute(0, 2, 1)  # [B, N, num_outputs]

        return prediction
