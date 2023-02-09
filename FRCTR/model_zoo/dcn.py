# -*- coding: UTF-8 -*-
"""
@project:RefineCTR
"""
import torch
import torch.nn as nn

from FRCTR.common import FeaturesEmbedding, MultiLayerPerceptron, CrossNetwork, BasicFRCTR

class CNFrn(BasicFRCTR):
    def __init__(self, field_dims, embed_dim, FRN=None, cn_layers=2):
        super(CNFrn, self).__init__(field_dims, embed_dim, FRN)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.cross_net = CrossNetwork(self.embed_output_dim, cn_layers)
        self.fc = nn.Linear(self.embed_output_dim, 1)

    def forward(self, x):
        x_emb = self.embedding(x)
        x_emb, _ = self.frn(x_emb)
        x_emb = x_emb.reshape(-1, self.embed_output_dim)
        cross_cn = self.cross_net(x_emb)
        pred_y = self.fc(cross_cn)
        return pred_y

class DCNFrn(BasicFRCTR):
    def __init__(self, field_dims, embed_dim, FRN=None, cn_layers=3, mlp_layers=(400, 400, 400), dropout=0.5):
        super(DCNFrn, self).__init__(field_dims, embed_dim, FRN)

        self.embed_output_dim = len(field_dims) * embed_dim
        self.cross_net = CrossNetwork(self.embed_output_dim, cn_layers)
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_layers, output_layer=False, dropout=dropout)
        self.fc = nn.Linear(mlp_layers[-1] + self.embed_output_dim, 1)

    def forward(self, x):
        x_emb = self.embedding(x)
        x_emb, _ = self.frn(x_emb)
        x_emb = x_emb.reshape(-1, self.embed_output_dim)

        cross_cn = self.cross_net(x_emb)
        cross_mlp = self.mlp(x_emb)
        pred_y = self.fc(torch.cat([cross_cn, cross_mlp], dim=1))
        return pred_y


class DCNFrnP(nn.Module):
    def __init__(self, field_dims, embed_dim, FRN1=None, FRN2=None,
                 cn_layers=3, mlp_layers=(400, 400, 400), dropout=0.5):
        super(DCNFrnP, self).__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim

        if not FRN1 or not FRN2:
            raise ValueError("Feature Refinement Network is None")
        self.frn1 = FRN1
        self.frn2 = FRN2

        self.cross_net = CrossNetwork(self.embed_output_dim, cn_layers)
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_layers, output_layer=False, dropout=dropout)
        self.fc = nn.Linear(mlp_layers[-1] + self.embed_output_dim, 1)

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



class DCN(nn.Module):
    def __init__(self, field_dims, embed_dim, cn_layers=3, mlp_layers=(400, 400, 400), dropout=0.5):
        super(DCN, self).__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim

        self.cross_net = CrossNetwork(self.embed_output_dim, cn_layers)
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_layers, output_layer=False, dropout=dropout)
        self.fc = nn.Linear(mlp_layers[-1] + self.input_dim, 1)

    def forward(self, x):
        x_emb = self.embedding(x)
        x_emb = self.frn(x_emb)
        x_emb = x_emb.reshape(-1, self.embed_output_dim)

        cross_cn = self.cross_net(x_emb)
        cross_mlp = self.mlp(x_emb)
        pred_y = self.fc(torch.cat([cross_cn, cross_mlp], dim=1))
        return pred_y
