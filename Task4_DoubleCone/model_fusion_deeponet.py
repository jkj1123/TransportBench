"""
Fusion-DeepONet: DeepONet with fused branch/trunk skip-connections.

Ported from Task3_Cavity/model_fusion_deeponet.py: architecture (branch_net,
trunk_net, the fusion parameters ab/cb/a1b/... and the skip-connection logic)
is untouched, only forward() is adapted to take a [B, C, H, W] image (channels
0:2 = shared (x, y) coordinates, channels 2:5 = branch/sensor scalars) instead
of separate (x_branch, x_trunk) tensors, and to return [B, num_outputs, H, W].
"""

import torch
import torch.nn as nn


class sin_act(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class Fusion_DeepONet(nn.Module):
    def __init__(self, branch_dim=1, trunk_dim=2, hidden_dim=280, num_outputs=4, depth=5, activation='GELU'):
        """
        Fusion-DeepONet.
        Args:
            branch_dim: sensor/parameter input dim
            trunk_dim: 2 (x, y)
            hidden_dim: hidden width
            num_outputs: number of output channels
        """
        super().__init__()

        if activation == 'GELU':
            self.act = nn.GELU()
        elif activation == 'Tanh':
            self.act = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.act2 = sin_act()

        #Fusion-deeponet additional parameters
        self.L = depth + 1

        self.ab  = nn.Parameter(torch.full((self.L,),0.1))
        self.cb  = nn.Parameter(torch.full((self.L,),0.1))
        self.a1b = nn.Parameter(torch.zeros(self.L,))
        self.F1b = nn.Parameter(torch.full((self.L,),0.1))
        self.c1b = nn.Parameter(torch.zeros(self.L,))

        self.at  = nn.Parameter(torch.full((self.L,),0.1))
        self.ct  = nn.Parameter(torch.full((self.L,),0.1))
        self.a1t = nn.Parameter(torch.zeros(self.L,))
        self.F1t = nn.Parameter(torch.full((self.L,),0.1))
        self.c1t = nn.Parameter(torch.zeros(self.L,))

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
        x_branch = x[:, 2:5, 0, 0]                              # [B, branch_dim]
        x_trunk = x[0, 0:2, :, :].permute(1, 2, 0).reshape(-1, 2)  # [H*W, 2]

        if x_trunk.dim() == 2:
            # [N, trunk_dim] -> [1, N, trunk_dim] -> [B, N, trunk_dim]
            x_trunk = x_trunk.unsqueeze(0).repeat(x_branch.shape[0], 1, 1)

        skip = []

        skip = []

        for i in range(self.L-1):

            x_branch = self.act(10*self.ab[i]*self.branch_net[i](x_branch)+self.cb[i])+\
            10*self.a1b[i]*self.act2(10*self.F1b[i]*self.branch_net[i](x_branch)+self.c1b[i])
            skip.append(x_branch)

        for i in range(1,self.L-1):
            skip[i] = skip[i-1]+skip[i]

        for i in range(self.L-1):

            x_trunk = self.act(10*self.at[i]*self.trunk_net[i](x_trunk)+self.ct[i])+\
            10*self.a1t[i]*self.act2(10*self.F1t[i]*self.trunk_net[i](x_trunk)+self.c1t[i])

            x_trunk = torch.einsum('bk,bik->bik', skip[i], x_trunk)

        x_branch = self.branch_net[-1](x_branch)
        x_trunk = self.trunk_net[-1](x_trunk)

        B_out_reshaped = x_branch.view(-1, self.num_outputs, self.hidden_dim)
        # [Batch, num_outputs, hidden_dim]

        prediction = torch.einsum("bnk, bik -> bin", B_out_reshaped, x_trunk)
        # [Batch, N_points, num_outputs]

        prediction = prediction.permute(0, 2, 1)
        prediction = prediction.reshape(B, self.num_outputs, H, W)

        return prediction
