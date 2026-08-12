"""
c_HyperDeepONet: DeepONet with a chunked hypernetwork trunk.

Key ideas from Lee & Shin:
- Branch net outputs ARE the trunk net's weights/biases (hypernetwork).
- No learned parameters in the trunk — all trunk params come from the branch output.
- The branch net is split into `num_chunk` passes (chunked hypernetwork) to keep
  the branch output layer small when the trunk needs many parameters.

Ported from Task3_Cavity/model_c_hyperdeeponet.py: architecture is untouched,
only forward() is adapted to take a [B, C, H, W] image (channels 0:2 = shared
(x, y) coordinates, channels 2:5 = branch/sensor scalars) instead of separate
(x_branch, x_trunk) tensors, and to return [B, num_outputs, H, W].
"""

import torch
import torch.nn as nn
import math


class c_HyperDeepONet(nn.Module):
    def __init__(self, branch_dim=674, trunk_dim=2, hidden_dim=46,
                 num_outputs=4, trunk_depth=3, branch_depth=3,
                 activation='GELU', num_chunk=1):
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
        self.trunk_dims = [trunk_dim] + [hidden_dim] * trunk_depth + [num_outputs]

        # Total parameters needed to construct the trunk net
        t_para = 0
        for i in range(len(self.trunk_dims) - 1):
            t_para += self.trunk_dims[i] * self.trunk_dims[i + 1] + self.trunk_dims[i + 1]

        ## defining number of chunks and number of sensors
        self.param_size = t_para
        self.num_chunks = num_chunk
        self.num_sensor = math.ceil(t_para / self.num_chunks)
        self.num_outputs = num_outputs

        # Branch: single network → t_para (trunk weights/biases)
        branch_dims = [branch_dim + 1] + [hidden_dim] * branch_depth + [self.num_sensor]
        self.branch_net = _MLP(branch_dims, act)

    def _branch_forward(self, x):
        """Run branch net on x → trunk parameters."""
        return self.branch_net(x)  # [B, t_para]

    def _trunk_forward(self, params, x_trunk):
        """Hypernetwork trunk: params → weights/biases → forward pass."""
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

        param_size = self.param_size
        num_chunk = self.num_chunks
        num_sensor = self.num_sensor

        params = x_branch.new_empty(B, num_chunk, num_sensor).uniform_(0, 1)    # [B, num_chunk, num_sensor]

        x_branch = x_branch.unsqueeze(1).repeat(1, num_chunk, 1)    # [B, branch_dim] -> [B, num_chunk, branch_dim]
        new_col = x_branch.new_zeros(B, num_chunk, 1)               # [B, num_chunk, 1]

        for i in range(num_chunk):
            new_col[:, i, 0] = i / num_chunk

        x_branch = torch.cat([x_branch, new_col], dim=2)     # [B, num_chunk, branch_dim + 1]

        params = self._branch_forward(x_branch)              # [B, num_chunk, num_sensor]

        params = params.reshape(B, num_chunk * num_sensor)   # [B, num_chunk * num_sensor]
        params = params[:, :param_size]                      # [B, param_size]

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
