# -*- coding: UTF-8 -*-
"""
@project:RefineCTR
"""

import torch.nn as nn

from FRCTR.common import FeaturesLinear, FeaturesEmbedding, MultiLayerPerceptron, BasicFRCTR

class FNN(nn.Module):
    def __init__(self, field_dims, embed_dim, mlp_layers=(400, 400, 400), dropout=0.5):
        super(FNN, self).__init__()
        self.lr = FeaturesLinear(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)

        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, embed_dims=mlp_layers,
                                        dropout=dropout, output_layer=True)

    def forward(self, x):
        x_emb = self.embedding(x)
        pred_y = self.lr(x) + self.mlp(x_emb.view(x.size(0), -1))
        return pred_y

class FNNFrn(BasicFRCTR):
    def __init__(self, field_dims, embed_dim, FRN=None,
                 mlp_layers=(400, 400, 400), dropout=0.5):
        super(FNN, self).__init__(field_dims, embed_dim, FRN)
        self.lr = FeaturesLinear(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)

        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, embed_dims=mlp_layers,
                                        dropout=dropout, output_layer=True)

    def forward(self, x):
        x_emb = self.embedding(x)
        x_emb, weight = self.frn(x_emb)
        pred_y = self.lr(x) + self.mlp(x_emb.view(x.size(0), -1))
        return pred_y