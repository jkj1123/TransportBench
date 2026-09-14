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
    def __init__(self, branch_dim=674, trunk_dim=2, hidden_dim=46, num_basis = 100,
                 num_outputs=4, trunk_depth=3, branch_depth=3,
                 activation='GELU',chunk_in= 100, chunk_out = 100):
        super().__init__()

        if activation == 'Tanh':
            act = nn.Tanh
            self._trunk_act = torch.tanh
        elif activation == 'GELU':
            act = nn.GELU
            self._trunk_act = nn.functional.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Trunk architecture: [trunk_dim, hidden, ..., hidden, num_outputs].
        # num_basis=None skips that extra hidden layer entirely -- the trunk
        # goes straight from the last hidden_dim layer to num_outputs.
        if num_basis is None:
            self.trunk_dims = [trunk_dim] + [hidden_dim] * trunk_depth + [num_outputs]
        else:
            self.trunk_dims = [trunk_dim] + [hidden_dim] * trunk_depth + [num_basis, num_outputs]

        # Total parameters needed to construct the trunk net


        ## defining number of chunks and number of sensors
        self.chunk_in = chunk_in
        self.chunk_out = chunk_out

        self.num_chunks = hidden_dim + 1

        self.latent_chunk = nn.Parameter(torch.randn(self.num_chunks, chunk_in))

        # Branch: single network → t_para (trunk weights/biases)
        self.branch_dims = [branch_dim + chunk_in] + [hidden_dim] * branch_depth + [chunk_out]
        self.trunk_dims = [trunk_dim] + [hidden_dim] * branch_depth + [num_outputs]
        self.branch_net = _MLP(self.branch_dims, act)
        self.trunk_net = _trunk_MLP(self.trunk_dims, act)

    def _branch_forward(self, x):
        """Run branch net on x → trunk parameters."""
        return self.branch_net(x)  # [B, t_para]

    def _branch_forward_part(self, x, i):
            """Run branch net on x → trunk parameters."""
            return self.branch_net.net[i](x)  # [B, t_para]

    def _trunk_forward(self, params, x_trunk):
        """Hypernetwork trunk: params → weights/biases → forward pass."""
        B = params[0].shape[0]  # use branch batch size (handles shared trunk)
        # Normalize to 3D: [B, N, trunk_dim]
        if x_trunk.dim() == 2:
            x_trunk = x_trunk.unsqueeze(0).expand(B, -1, -1)

        _, N, _ = x_trunk.shape
        y = x_trunk  # [B, N, trunk_dim]

        y = self.trunk_net.net[0](y)
        y = self.trunk_net.net[1](y)

        for i in range(1, len(self.trunk_dims) - 2):
            d_in, d_out = self.trunk_dims[i], self.trunk_dims[i + 1]

            weight = params[i][:,:-1,:]
            bias = params[i][:,-1:,:].reshape(B, 1, d_out)

            y = torch.einsum("bij,bgj->bgi", weight, y) + bias  # [B, N, d_out]
            y = self._trunk_act(y)

        # Last layer: no activation
        y = self.trunk_net.net[-1](y)
        return y

    def forward(self, x_branch, x_trunk):

        """
        Args:
            x_branch: [B, branch_dim]  sensor values
            x_trunk:  [B, N, trunk_dim]  query coordinates

        Returns:
            [B, N, num_outputs]
        """

        B = x_branch.shape[0]
        K = self.num_chunks
        
        x_branch = x_branch.unsqueeze(1).repeat(1, K, 1)    # [B, branch_dim] -> [B, 1, branch_dim] -> [B, K, branch_dim]
        z = self.latent_chunk.unsqueeze(0).expand(B, -1, -1)    # [K, chunk_in] -> [B, K, chunk_in]

        hyper_input = torch.cat([x_branch, z], dim=-1)    # [B, K, branch_dim + chunk_in]

        params_list = []

        #for i in range(len(self.branch_dims) - 2):
        #    hyper_input = self._branch_forward_part(hyper_input, 2*i) # [B, K, branch_width]
        #    hyper_input = self._branch_forward_part(hyper_input, 2*i+1) # [B, K, branch_width]
        #    params_list.append(hyper_input)

        for i in range(len(self.branch_dims) - 2):
            # Linear
            h_lin = self._branch_forward_part(hyper_input, 2*i)

            # raw Linear output을 trunk weight/bias로 사용
            params_list.append(h_lin)

            # branch network 자체는 activation을 거쳐 계속 진행
            hyper_input = self._branch_forward_part(h_lin, 2*i+1)
        
        #params = params.reshape(B, -1)     # [B, K * chunk_out]
        #params = params[:, :self.param_size]                     # [B, param_size]

        return self._trunk_forward(params_list, x_trunk)


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

class _trunk_MLP(nn.Module):
    """Simple fully-connected stack: Linear → Act → ... → Linear."""
    def __init__(self, dims, act):
        super().__init__()
        layers = []
        layers.append(nn.Linear(dims[0], dims[1]))
        layers.append(act())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)    

    
