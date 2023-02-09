# -*- coding: UTF-8 -*-
"""
@project:RefineCTR
"""

import torch.nn as nn

from FRCTR.common import FeaturesLinear, FeaturesEmbedding, BasicFRCTR, FactorizationMachine


class FM(nn.Module):
    def __init__(self, field_dims, emb_dim):
        super(FM, self).__init__()
        self.lr = FeaturesLinear(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, emb_dim)
        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, x):
        x_emb = self.embedding(x)
        pred_y = self.lr(x) + self.fm(x_emb)
        return pred_y


class FMFrn(BasicFRCTR):
    def __init__(self, field_dims, emb_dim, FRN=None):
        super().__init__(field_dims, emb_dim, FRN)
        self.lr = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, x):
        x_emb = self.embedding(x)
        x_emb, _ = self.frn(x_emb)
        pred_y = self.lr(x) + self.fm(x_emb)
        return pred_y
