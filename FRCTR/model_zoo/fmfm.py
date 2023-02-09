# -*- coding: UTF-8 -*-
"""
@project:RefineCTR
"""

import torch
import torch.nn as nn

from FRCTR.common import FeaturesLinear, FactorizationMachine, FeaturesEmbedding


class FMFM(nn.Module):
    def __init__(self, field_dims, embed_dim, interaction_type="matrix"):
        super(FMFM, self).__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.lr = FeaturesLinear(field_dims)
        self.num_field = len(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.inter_num = self.num_field * (self.num_field - 1) // 2
        self.field_interaction_type = interaction_type
        if self.field_interaction_type == "vector":  # FvFM
            # F,1, E
            self.interaction_weight = nn.Parameter(torch.Tensor(self.inter_num, embed_dim))
        elif self.field_interaction_type == "matrix":  # FmFM
            # F,E,E
            self.interaction_weight = nn.Parameter(torch.Tensor(self.inter_num, embed_dim, embed_dim))
        nn.init.xavier_uniform_(self.interaction_weight.data)
        # self.triu_index = torch.triu(torch.ones(self.num_field, self.num_field), 1).nonzero().cuda()
        self.row, self.col = list(), list()
        for i in range(self.num_field - 1):
            for j in range(i + 1, self.num_field):
                self.row.append(i), self.col.append(j)

    def forward(self, x):
        x_emb = self.embedding(x)  # (B,F, E)
        # left_emb = torch.index_select(emb_x, 1, self.triu_index[:, 0]).cuda() #B,F,E
        # right_emb = torch.index_select(emb_x, 1, self.triu_index[:, 1]).cuda() # B,F,E
        left_emb = x_emb[:, self.row]
        right_emb = x_emb[:, self.col]
        #  Transfer the embedding space of left_emb to corresponding space
        if self.field_interaction_type == "vector":
            left_emb = left_emb * self.interaction_weight  # B,I,E
        elif self.field_interaction_type == "matrix":
            # B,F,1,E * F,E,E = B,F,1,E => B,F,E
            left_emb = torch.matmul(left_emb.unsqueeze(2), self.interaction_weight).squeeze(2)
        # FM interaction
        pred_y = (left_emb * right_emb).sum(dim=-1).sum(dim=-1, keepdim=True)
        pred_y += self.lr(x)
        return pred_y
