# -*- coding: UTF-8 -*-
"""
@project:RefineCTR
"""

import torch

from torch import nn


class FWNLayer(nn.Module):
    #  Also known as field-wise network: FWN
    def __init__(self, field_length, embed_dim):
        super(FWNLayer, self).__init__()
        self.input_dim = field_length * embed_dim
        self.local_w = nn.Parameter(torch.randn(field_length, embed_dim, embed_dim))
        self.local_b = nn.Parameter(torch.randn(field_length, 1, embed_dim))

        nn.init.xavier_uniform_(self.local_w.data)
        nn.init.xavier_uniform_(self.local_b.data)

    def forward(self, x_emb):
        x_local = torch.matmul(x_emb.permute(1, 0, 2), self.local_w) + self.local_b
        x_local0 = torch.relu(x_local).permute(1, 0, 2)
        x_local = x_local0 * x_emb
        return x_local.contiguous(), x_local0
