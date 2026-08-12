"""
c_HyperDeepONet: DeepONet with a chunked hypernetwork trunk.

Key ideas from Lee & Shin:
- Branch net outputs ARE the trunk net's weights/biases (hypernetwork).
- No learned parameters in the trunk — all trunk params come from the branch output.
- The branch net is split into `num_chunks` passes over a learned latent code
  (chunked hypernetwork) to keep the branch output layer small when the trunk
  needs many parameters.

Ported from Task3_Cavity/model_c_hyperdeeponet.py: architecture is untouched,
only forward() is adapted to take a [B, C, H, W] image (channels 0:2 = shared
(x, y) coordinates, channels 2:5 = branch/sensor scalars) instead of separate
(x_branch, x_trunk) tensors, and to return [B, num_outputs, H, W].
"""

import torch
import torch.nn as nn
import math


class c_HyperDeepONet(nn.Module):
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

        # Trunk architecture: [trunk_dim, hidden, ..., hidden, num_outputs]
        self.trunk_dims = ([trunk_dim] + [hidden_dim] * trunk_depth + [num_basis, num_outputs])

        # Total parameters needed to construct the trunk net
        t_para = 0
        for i in range(len(self.trunk_dims) - 1):
            t_para += self.trunk_dims[i] * self.trunk_dims[i + 1] + self.trunk_dims[i + 1]


        ## defining number of chunks and number of sensors
        self.param_size = t_para
        self.chunk_in = chunk_in
        self.chunk_out = chunk_out
        self.num_outputs = num_outputs

        self.num_chunks = math.ceil(self.param_size / chunk_out)

        self.latent_chunk = nn.Parameter(torch.randn(self.num_chunks, chunk_in))

        # Branch: single network → t_para (trunk weights/biases)
        branch_dims = [branch_dim + chunk_in] + [hidden_dim] * branch_depth + [chunk_out]
        self.branch_net = _MLP(branch_dims, act)

    def _branch_forward(self, x):
        """Run branch net on x → trunk parameters."""
        return self.branch_net(x)  # [B, t_para]

    def _trunk_forward(self, params, x_trunk):
        """Hypernetwork trunk: params → weights/biases → forward pass."""
        # Normalize to 3D: [B, N, trunk_dim]
        if x_trunk.dim() == 2:
            x_trunk = x_trunk.unsqueeze(0)  # [1, N, trunk_dim]

        B, N, _ = x_trunk.shape
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

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] image tensor.
               channels 0:2 -> (x, y) coordinates, shared across the batch
               channels 2:5 -> branch/sensor scalars (constant over H, W)

        Returns:
            [B, num_outputs, H, W]
        """

        B, C, H, W = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        x_branch = x[:, 2:5, 0, 0]                                       # [B, 3]
        x_trunk = x[0:1, 0:2, :, :].permute(0, 2, 3, 1)                  # [1, H, W, 2]
        x_trunk = x_trunk.reshape(x_trunk.shape[0], -1, x_trunk.shape[-1])  # [1, H*W, 2]
        x_trunk = x_trunk.expand(B, -1, -1)                              # [1, H*W, 2] -> [B, H*W, 2]

        K = self.num_chunks

        x_branch = x_branch.unsqueeze(1).repeat(1, K, 1)    # [B, branch_dim] -> [B, 1, branch_dim] -> [B, K, branch_dim]
        z = self.latent_chunk.unsqueeze(0).expand(B, -1, -1)    # [K, chunk_in] -> [B, K, chunk_in]

        hyper_input = torch.cat([x_branch, z], dim=-1)    # [B, K, branch_dim + chunk_in]

        params = self._branch_forward(hyper_input)    # [B, K, chunk_out]

        params = params.reshape(B, -1)     # [B, K * chunk_out]
        params = params[:, :self.param_size]                     # [B, param_size]

        output = self._trunk_forward(params, x_trunk)        # [B, N, num_outputs]
        output = output.permute(0, 2, 1)
        output = output.reshape(B, self.num_outputs, H, W)

        return output


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
