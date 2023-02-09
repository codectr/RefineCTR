# -*- coding: UTF-8 -*-
"""
@project:RefineCTR
"""

import torch
import torch.nn as nn

from FRCTR.common import FeaturesEmbedding, MultiLayerPerceptron, CrossNetworkV2, BasicFRCTR

class CrossNetV2(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, cn_layers=3):
        super(CrossNetV2, self).__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.cross_net = CrossNetworkV2(self.embed_output_dim, cn_layers)
        self.pred_layer = torch.nn.Linear(self.embed_output_dim, 1)

    def forward(self, x):
        x_embed = self.embedding(x).view(-1, self.embed_output_dim)
        cross_cn = self.cross_net(x_embed)
        pred_y = self.pred_layer(cross_cn)
        return pred_y


class CN2Frn(BasicFRCTR):
    def __init__(self, field_dims, embed_dim, FRN=None, cn_layers=3):
        super(CN2Frn, self).__init__(field_dims, embed_dim, FRN)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.cross_net = CrossNetworkV2(self.embed_output_dim, cn_layers)
        self.pred_layer = torch.nn.Linear(self.embed_output_dim, 1)

    def forward(self, x):
        x_emb = self.embedding(x)
        x_emb, _ = self.frn(x_emb)
        x_emb = x_emb.reshape(-1, self.embed_output_dim)
        cross_cn = self.cross_net(x_emb)
        pred_y = self.pred_layer(cross_cn)
        return pred_y

class DCNV2(nn.Module):
    def __init__(self, field_dims, embed_dim, cn_layers=3, mlp_layers=(400, 400, 400), dropout=0.5):
        super(DCNV2, self).__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.cross_net = CrossNetworkV2(self.embed_output_dim, cn_layers)
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_layers, output_layer=False, dropout=dropout)
        self.fc = torch.nn.Linear(mlp_layers[-1] + self.embed_output_dim, 1)

    def forward(self, x):
        x_emb = self.embedding(x).view(-1, self.embed_output_dim)  # B,F*E
        cross_cn = self.cross_net(x_emb)
        cross_mlp = self.mlp(x_emb)

        pred_y = self.fc(torch.cat([cross_cn, cross_mlp], dim=1))
        return pred_y


class DCNV2Frn(BasicFRCTR):
    def __init__(self, field_dims, embed_dim, FRN=None, cn_layers=3,
                 mlp_layers=(400, 400, 400), dropout=0.5):
        super(DCNV2Frn, self).__init__(field_dims, embed_dim, FRN)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.cross_net = CrossNetworkV2(self.embed_output_dim, cn_layers)
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_layers, output_layer=False, dropout=dropout)
        self.fc = torch.nn.Linear(mlp_layers[-1] + self.embed_output_dim, 1)

    def forward(self, x):
        x_emb = self.embedding(x)
        x_emb, _ = self.frn(x_emb)
        x_emb = x_emb.reshape(-1, self.embed_output_dim)
        cross_cn = self.cross_net(x_emb)
        cross_mlp = self.mlp(x_emb)

        pred_y = self.fc(torch.cat([cross_cn, cross_mlp], dim=1))
        return pred_y


class DCNV2FrnP(nn.Module):
    def __init__(self, field_dims, embed_dim, FRN1=None, FRN2=None,
                 cn_layers=4, mlp_layers=(400, 400, 400),
                 dropout=0.5):
        super(DCNV2FrnP, self).__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        if not FRN1 or not FRN2:
            raise ValueError("Feature Refinement Network is None")
        self.frn1 = FRN1
        self.frn2 = FRN2
        self.embed_output_dim = len(field_dims) * embed_dim
        self.cross_net = CrossNetworkV2(self.embed_output_dim, cn_layers)
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_layers, output_layer=False, dropout=dropout)
        self.fc = torch.nn.Linear(mlp_layers[-1] + self.embed_output_dim, 1)

    def forward(self, x):
        x_emb = self.embedding(x)
        x_emb1, _ = self.frn1(x_emb)
        x_emb2, _ = self.frn2(x_emb)
        x_emb1 = x_emb1.reshape(-1, self.embed_output_dim)
        x_emb2 = x_emb2.reshape(-1, self.embed_output_dim)
        cross_cn = self.cross_net(x_emb1)
        cross_mlp = self.mlp(x_emb2)

        pred_y = self.fc(torch.cat([cross_cn, cross_mlp], dim=1))
        return pred_y


class DCNV2S(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, cn_layers=3, mlp_layers=(400, 400, 400), dropout=0.5):
        super(DCNV2S, self).__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)

        self.embed_output_dim = len(field_dims) * embed_dim
        self.cross_net = CrossNetworkV2(self.embed_output_dim, cn_layers)
        self.pred_layer = MultiLayerPerceptron(self.embed_output_dim, mlp_layers, output_layer=True,
                                               dropout=dropout)

    def forward(self, x):
        x_embed = self.embedding(x).view(-1, self.embed_output_dim)
        cross_cn = self.cross_net(x_embed)
        pred_y = self.pred_layer(cross_cn)
        return pred_y
