import torch
import torch.nn as nn

class sin_act(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class Bilinear_DeepONet(nn.Module):
    def __init__(self, branch_dim=1, trunk_dim=2, hidden_dim=280, num_outputs=10, depth=5, activation='GELU'):
        """
        DeepONet for Cavity Task (Square Domain)
        Args:
            branch_dim: 1
            trunk_dim: 2 (x, y)
            hidden_dim: 280 (Target ~1.0M Params)
            num_outputs: 10 (Physics quantities)
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
        self.r = hidden_dim

        self.ab  = nn.Parameter(torch.full((self.L,),0.1))
        self.cb  = nn.Parameter(torch.full((self.L,),0.1))
        self.a1b = nn.Parameter(torch.zeros(self.L,))
        self.F1b = nn.Parameter(torch.full((self.L,),0.1))
        self.c1b = nn.Parameter(torch.zeros(self.L,))

        self.U = nn.Parameter(
            torch.eye(hidden_dim).unsqueeze(0).repeat(self.L, 1, 1)
        )

        self.V = nn.Parameter(
            torch.eye(hidden_dim).unsqueeze(0).repeat(self.L, 1, 1)
        )

        self.register_buffer("C", torch.eye(hidden_dim))
        # Fixed (non-learnable), shared across all L layers -- register as a
        # buffer (not a bare attribute) so it moves with .to(device) and is
        # included in state_dict.


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

    def forward(self, x_branch, x_trunk):

        """Forward pass.

        Args:
            x_branch: [Batch, branch_dim]
            x_trunk:  [N_points, trunk_dim]

        Returns:
            [Batch, N_points, num_outputs]
        """

        if x_trunk.dim() == 2:
            # [N, trunk_dim] -> [1, N, trunk_dim] -> [B, N, trunk_dim]
            x_trunk = x_trunk.unsqueeze(0).repeat(x_branch.shape[0], 1, 1)

        skip = []
        reduced_skip = []

        for i in range(self.L - 1):
            branch_lin = self.branch_net[i](x_branch)
            x_branch = self.act(branch_lin)
            skip.append(x_branch)

        for i in range(1, self.L - 1):
            skip[i] = skip[i-1] + skip[i]

        reduced_skip = [
            skip[i] @ self.U[i]
            for i in range(self.L - 1)
        ]

        for i in range(self.L-1):

            trunk_lin = self.trunk_net[i](x_trunk)
            x_trunk = self.act(trunk_lin)

            x_reduced_trunk = torch.einsum('bik,kr->bir', x_trunk, self.V[i,:,:])

            x_reduced_trunk = torch.einsum('br,bir->bir', reduced_skip[i], x_reduced_trunk)
            x_trunk = torch.einsum('rk,bir->bik', self.C, x_reduced_trunk)

        x_branch = self.branch_net[-1](x_branch)
        x_trunk = self.trunk_net[-1](x_trunk)

        B_out_reshaped = x_branch.view(-1, self.num_outputs, self.hidden_dim)
        # [Batch, num_outputs, hidden_dim]

        prediction = torch.einsum("bnk, bik -> bin", B_out_reshaped, x_trunk)
        # [Batch, N_points, num_outputs]

        return prediction

