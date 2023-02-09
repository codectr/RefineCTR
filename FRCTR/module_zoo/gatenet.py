# -*- coding: UTF-8 -*-
"""
@project:RefineCTR
"""

import torch
from torch import nn


class GateLayer(nn.Module):
    def __init__(self, field_length, embed_dim, gate_type="vec"):
        super(GateLayer, self).__init__()
        if gate_type == "bit":
            self.local_w = nn.Parameter(torch.randn(field_length, embed_dim, embed_dim))
        elif gate_type == "vec":
            self.local_w = nn.Parameter(torch.randn(field_length, embed_dim, 1))
        nn.init.xavier_uniform_(self.local_w.data)

    def forward(self, x_emb):
        x_weight = torch.matmul(x_emb.permute(1, 0, 2), self.local_w)
        x_weight = x_weight.permute(1, 0, 2)
        x_emb_weight = x_weight * x_emb
        return x_emb_weight.contiguous(), x_weight
