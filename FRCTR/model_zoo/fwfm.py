# -*- coding: UTF-8 -*-
"""
@project:RefineCTR
"""

import torch
import torch.nn as nn

from FRCTR.common import FeaturesLinear, FeaturesEmbedding, BasicFRCTR


class FwFM(nn.Module):
    def __init__(self, field_dims, embed_dim):
        super(FwFM, self).__init__()
        self.lr = FeaturesLinear(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.fwfm = FwFMInterLayer(len(field_dims))

    def forward(self, x):
        x_emb = self.embedding(x)
        pred_y = self.lr(x) + self.fwfm(x_emb)
        return pred_y


class FwFMInterLayer(nn.Module):
    def __init__(self, num_fields):
        super(FwFMInterLayer, self).__init__()

        self.num_fields = num_fields
        num_inter = (num_fields * (num_fields - 1)) // 2

        self.fc = nn.Linear(num_inter, 1)
        self.row, self.col = list(), list()
        for i in range(self.num_fields - 1):
            for j in range(i + 1, self.num_fields):
                self.row.append(i), self.col.append(j)

    def forward(self, x_embed):
        x_inter = torch.sum(x_embed[:, self.row] * x_embed[:, self.col], dim=2, keepdim=False)
        inter_sum = self.fc(x_inter)
        return inter_sum


class FwFMFrn(BasicFRCTR):
    def __init__(self, field_dims, embed_dim, FRN=None):
        super(FwFMFrn, self).__init__(field_dims, embed_dim, FRN)
        self.lr = FeaturesLinear(field_dims)
        self.fwfm = FwFMInterLayer(len(field_dims))

    def forward(self, x):
        x_emb = self.embedding(x)
        x_emb, weight = self.frn(x_emb)
        pred_y = self.lr(x) + self.fwfm(x_emb)
        return pred_y
