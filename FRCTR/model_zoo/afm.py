# -*- coding: UTF-8 -*-
"""
@project:RefineCTR
"""

import torch

from FRCTR.common import FeaturesLinear, FeaturesEmbedding, AttentionalFactorizationMachine, BasicFRCTR


class AFM(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, attn_size, dropouts=(0.5, 0.5)):
        super().__init__()
        self.num_fields = len(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        self.afm = AttentionalFactorizationMachine(embed_dim, attn_size,
                                                   self.num_fields, dropouts=dropouts,
                                                   reduce=True)

    def forward(self, x):
        x_emb = self.embedding(x)
        cross_term = self.afm(x_emb)
        pred_y = self.linear(x) + cross_term
        return pred_y


class AFMFrn(BasicFRCTR):
    def __init__(self, field_dims, embed_dim, FRN=None, attn_size=16, dropouts=(0.5, 0.5)):
        super().__init__(field_dims, embed_dim, FRN)
        self.num_fields = len(field_dims)
        self.linear = FeaturesLinear(field_dims)
        self.afm = AttentionalFactorizationMachine(embed_dim, attn_size,
                                                   self.num_fields, dropouts=dropouts,
                                                   reduce=True)

    def forward(self, x):
        x_emb = self.embedding(x)
        x_emb, weight = self.frn(x_emb)
        cross_term = self.afm(x_emb)
        pred_y = self.linear(x) + cross_term
        return pred_y
