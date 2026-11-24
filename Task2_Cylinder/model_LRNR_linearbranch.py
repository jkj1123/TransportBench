"""
Ablation of LRNR (low_rank_models/model_LRNR.py): replaces the branch net
-- a multi-layer MLP (Linear -> Act -> ... -> Linear) -- with a SINGLE
plain nn.Linear(branch_dim, t_para), no activation, no hidden layers.

    plain LRNR:        s_w_l, s_b_l = MLP(x_branch)      (GELU/Tanh hidden layers)
    this variant:      s_w_l, s_b_l = Linear(x_branch)   (no nonlinearity anywhere in the branch)

Everything else -- U_l, V_l, B_l, rank s, s_w_offset, activation (trunk
only), trunk_depth, first/last trunk layers, forward() -- is reused
unchanged from LRNR. t_para is read off the original branch MLP's own
final layer (`.out_features`) before replacing it, so this stays correct
for any trunk_depth/hidden_dim/s without recomputing the t_para formula
by hand.
"""

import torch.nn as nn

from model_LRNR import LRNR


class LRNR_LinearBranch(LRNR):
    def __init__(self, branch_dim=674, trunk_dim=2, hidden_dim=46,
                 num_outputs=4, trunk_depth=3, branch_depth=3,
                 activation='GELU', s=12, s_w_offset=0.0):
        super().__init__(branch_dim=branch_dim, trunk_dim=trunk_dim,
                          hidden_dim=hidden_dim, num_outputs=num_outputs,
                          trunk_depth=trunk_depth, branch_depth=branch_depth,
                          activation=activation, s=s, s_w_offset=s_w_offset)
        t_para = self.branch_net.net[-1].out_features  # read off the MLP's own output size
        self.branch_net = nn.Linear(branch_dim, t_para)
