# -*- coding: UTF-8 -*-
"""
@project:RefineCTR
"""

import torch
import torch.nn as nn

from FRCTR.common import FeaturesLinear, FeaturesEmbedding, FactorizationMachine, MultiLayerPerceptron, BasicFRCTR


class NeuralFactorizationMachineModel(nn.Module):
    def __init__(self, field_dims, embed_dim, mlp_dims, dropouts):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        self.fm = torch.nn.Sequential(
            FactorizationMachine(reduce_sum=False),
            torch.nn.BatchNorm1d(embed_dim),
            torch.nn.Dropout(dropouts[0])
        )
        self.mlp = MultiLayerPerceptron(embed_dim, mlp_dims, dropouts[1])

    def forward(self, x):
        x_emb = self.embedding(x)
        cross_term = self.fm(x_emb)
        pred_y = self.linear(x) + self.mlp(cross_term)
        return pred_y


class NFMFrn(BasicFRCTR):
    def __init__(self, field_dims, embed_dim, FRN=None, mlp_layers=(400, 400, 400), dropouts=(0.5, 0.5)):
        super().__init__(field_dims, embed_dim, FRN)
        self.linear = FeaturesLinear(field_dims)
        self.fm = torch.nn.Sequential(
            FactorizationMachine(reduce_sum=False),
            torch.nn.BatchNorm1d(embed_dim),
            torch.nn.Dropout(dropouts[0])
        )
        self.mlp = MultiLayerPerceptron(embed_dim, mlp_layers, dropouts[1])

    def forward(self, x):
        x_emb = self.embedding(x)
        x_emb, _ = self.frn(x_emb)
        cross_term = self.fm(x_emb)
        pred_y = self.linear(x) + self.mlp(cross_term)
        return pred_y
