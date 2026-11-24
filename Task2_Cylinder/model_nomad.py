"""
NOMAD: Nonlinear Manifold Decoders for Operator Learning
(Seidman et al., 2022, https://arxiv.org/abs/2206.03551) -- minimal baseline.

    beta(mu) = B_theta(mu) in R^r                  (Branch MLP -> latent vector)
    u_hat(x; mu) = D_phi([beta(mu), x])             (Decoder MLP on concatenated input)

Branch and Decoder are both plain MLPs with ordinary, input-independent
`nn.Parameter` weights. The Branch only ever produces a latent vector --
it never generates Decoder weights (no hypernetwork), there is no
branch-trunk dot product (as in DeepONet/HyperDeepONet), no low-rank
weight factorization, no FiLM, no residual connections.

Matches HyperDeepONet's constructor signature and forward() interface
(model_hyperdeeponet.py) plus one extra argument, `latent_dim` (the size
of beta), so it drops into the same training/eval code. No input
normalization or output denormalization is added here -- x_branch and
x_trunk are consumed exactly as the existing data pipeline provides them
(see model_coordinate_mlp.py's docstring for the Task2_Cylinder specifics).
"""

import torch
import torch.nn as nn


class NOMAD(nn.Module):
    def __init__(self, branch_dim=674, trunk_dim=2, hidden_dim=46,
                 num_outputs=4, trunk_depth=3, branch_depth=3,
                 activation='GELU', latent_dim=16):
        super().__init__()

        if activation == 'GELU':
            act = nn.GELU
        elif activation == 'Tanh':
            act = nn.Tanh
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.branch_dim = branch_dim
        self.trunk_dim = trunk_dim
        self.latent_dim = latent_dim
        self.num_outputs = num_outputs

        # Branch MLP: branch_dim -> hidden (x branch_depth) -> latent_dim.
        # branch_depth is the number of hidden layers; no activation on the
        # final (latent-producing) layer.
        branch_dims = [branch_dim] + [hidden_dim] * branch_depth + [latent_dim]
        branch_layers = []
        for i in range(len(branch_dims) - 2):
            branch_layers.append(nn.Linear(branch_dims[i], branch_dims[i + 1]))
            branch_layers.append(act())
        branch_layers.append(nn.Linear(branch_dims[-2], branch_dims[-1]))
        self.branch_net = nn.Sequential(*branch_layers)

        # Decoder MLP: (latent_dim + trunk_dim) -> hidden (x trunk_depth) -> num_outputs.
        # trunk_depth is the number of hidden layers; no activation on the output layer.
        decoder_dims = [latent_dim + trunk_dim] + [hidden_dim] * trunk_depth + [num_outputs]
        decoder_layers = []
        for i in range(len(decoder_dims) - 2):
            decoder_layers.append(nn.Linear(decoder_dims[i], decoder_dims[i + 1]))
            decoder_layers.append(act())
        decoder_layers.append(nn.Linear(decoder_dims[-2], decoder_dims[-1]))
        self.decoder_net = nn.Sequential(*decoder_layers)

    def forward(self, x_branch, x_trunk):
        """
        Args:
            x_branch: [B, branch_dim]  condition vector (e.g. Kn, Ma)
            x_trunk:  [N, trunk_dim] (shared across the batch) or
                      [B, N, trunk_dim] (per-sample coordinates)

        Returns:
            [B, N, num_outputs]
        """
        B = x_branch.shape[0]
        beta = self.branch_net(x_branch)  # [B, latent_dim]

        if x_trunk.dim() == 2:
            N = x_trunk.shape[0]
            x_trunk = x_trunk.unsqueeze(0).expand(B, N, self.trunk_dim)
        else:
            N = x_trunk.shape[1]

        beta_exp = beta.unsqueeze(1).expand(B, N, self.latent_dim)
        x = torch.cat([beta_exp, x_trunk], dim=-1)  # [B, N, latent_dim + trunk_dim]
        return self.decoder_net(x)

    def count_params(self):
        """Trainable parameter counts, broken down by sub-network."""
        branch = sum(p.numel() for p in self.branch_net.parameters())
        decoder = sum(p.numel() for p in self.decoder_net.parameters())
        return {'branch': branch, 'decoder': decoder, 'total': branch + decoder}
