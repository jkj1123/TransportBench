"""
HyperDeepONet: DeepONet with a hypernetwork trunk.

Key ideas from Lee & Shin:
- Branch net outputs ARE the trunk net's weights/biases (hypernetwork).
- No learned parameters in the trunk — all trunk params come from the branch output.
"""

import torch
import torch.nn as nn
import math


class c_HyperDeepONet(nn.Module):
    def __init__(self, branch_dim=674, trunk_dim=2, hidden_dim=46,
                 num_outputs=4, trunk_depth=3, branch_depth=3,
                 activation='GELU',num_chunk = 1):
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

        # Branch: single network → t_para (trunk weights/biases)
        branch_dims = [branch_dim + 1] + [hidden_dim] * branch_depth + [self.num_sensor]
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

    def forward(self, x_branch, x_trunk):

        """
        Args:
            x_branch: [B, branch_dim]  sensor values
            x_trunk:  [N, trunk_dim] or [B, N, trunk_dim]  query coordinates

        Returns:
            [B, N, num_outputs]
        """

        B,D = x_branch.shape[0], x_branch.shape[1]

        param_size = self.param_size
        num_chunk = self.num_chunks
        num_sensor = self.num_sensor

        params = x_branch.new_empty(B, num_chunk, num_sensor).uniform_(0, 1)
        
        x_branch = x_branch.unsqueeze(1).repeat(1, num_chunk, 1)
        new_col = x_branch.new_zeros(B, num_chunk, 1)

        
        for i in range(num_chunk):
            new_col[:,i,0] = i/num_chunk
        
        x_branch = torch.cat([x_branch, new_col], dim=2)
        
        for i in range(num_chunk):
            params[:,:,:] =  self._branch_forward(x_branch[:,:,:])
        
        params = params.reshape(B,num_chunk*num_sensor)
        params = params[:, :param_size]

        return self._trunk_forward(params, x_trunk)


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
