# -*- coding: UTF-8 -*-
"""
@project:RefineCTR
"""

import torch
import torch.nn as nn

from FRCTR.common import FeaturesEmbedding, MultiLayerPerceptron, \
    FeaturesLinear, SenetLayer, BilinearInteractionLayer


class FiBiNet(nn.Module):
    def __init__(self, field_dims, embed_dim, mlp_layers=(400, 400, 400), dropout=0.5, bilinear_type="all"):
        super(FiBiNet, self).__init__()
        num_fields = len(field_dims)
        self.linear = FeaturesLinear(field_dims)

        self.embedding = FeaturesEmbedding(field_dims, embed_dim)

        self.senet = SenetLayer(num_fields)

        self.bilinear = BilinearInteractionLayer(num_fields, embed_dim, bilinear_type=bilinear_type)
        self.bilinear2 = BilinearInteractionLayer(num_fields, embed_dim, bilinear_type=bilinear_type)

        num_inter = num_fields * (num_fields - 1) // 2
        self.embed_output_size = num_inter * embed_dim
        self.mlp = MultiLayerPerceptron(2 * self.embed_output_size, mlp_layers, dropout=dropout)

    def forward(self, x):
        lin = self.linear(x)
        x_emb = self.embedding(x)
        x_senet, x_weight = self.senet(x_emb)

        x_bi1 = self.bilinear(x_emb)
        x_bi2 = self.bilinear2(x_senet)

        x_con = torch.cat([x_bi1.view(x.size(0), -1),
                           x_bi2.view(x.size(0), -1)], dim=1)

        pred_y = self.mlp(x_con) + lin
        return pred_y


class FiBiNetFrn(nn.Module):
    def __init__(self, field_dims, embed_dim, FRN1=None, FRN2=None,
                 mlp_layers=(400, 400, 400), dropout=0.5, bilinear_type="all"):
        super(FiBiNetFrn, self).__init__()
        num_fields = len(field_dims)
        self.linear = FeaturesLinear(field_dims)

        self.embedding = FeaturesEmbedding(field_dims, embed_dim)

        if not FRN1 or not FRN2:
            raise ValueError("Feature Refinement Network is None")
        self.frn1 = FRN1
        self.frn2 = FRN2

        self.bilinear = BilinearInteractionLayer(num_fields, embed_dim, bilinear_type=bilinear_type)
        self.bilinear2 = BilinearInteractionLayer(num_fields, embed_dim, bilinear_type=bilinear_type)

        num_inter = num_fields * (num_fields - 1) // 2
        self.embed_output_size = num_inter * embed_dim
        self.mlp = MultiLayerPerceptron(2 * self.embed_output_size, mlp_layers, dropout=dropout)

    def forward(self, x):
        lin = self.linear(x)
        x_emb = self.embedding(x)
        x_emb1, x_weight1 = self.frn1(x_emb)
        x_emb2, x_weight2 = self.frn2(x_emb)

        x_bi1 = self.bilinear(x_emb1)
        x_bi2 = self.bilinear2(x_emb2)

        x_con = torch.cat([x_bi1.view(x.size(0), -1),
                           x_bi2.view(x.size(0), -1)], dim=1)

        pred_y = self.mlp(x_con) + lin
        return pred_y
