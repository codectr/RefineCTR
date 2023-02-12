# -*- coding: UTF-8 -*-
"""
@project:RefineCTR
"""

import torch
import torch.nn as nn

from FRCTR.common import FeaturesEmbedding, InnerProductNetwork, OuterProductNetwork, MultiLayerPerceptron, BasicFRCTR
from common import OuterProductNetwork2


class IPNN(nn.Module):
    def __init__(self, field_dims, embed_dim, mlp_layers=(400, 400, 400), dropout=0.5):
        super(IPNN, self).__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        num_fields = len(field_dims)
        self.pnn = InnerProductNetwork(num_fields)

        self.embed_output_dim = num_fields * embed_dim
        self.inter_size = num_fields * (num_fields - 1) // 2
        self.mlp = MultiLayerPerceptron(self.inter_size + self.input_dim, mlp_layers, dropout=dropout)

    def forward(self, x):
        # B,F,E
        x_emb = self.embedding(x)
        cross_ipnn = self.pnn(x_emb)

        x = torch.cat([x_emb.view(-1, self.embed_output_dim), cross_ipnn], dim=1)
        pred_y = self.mlp(x)
        return pred_y


class IPNNFrn(BasicFRCTR):
    def __init__(self, field_dims, embed_dim, FRN=None, mlp_layers=(400, 400, 400), dropout=0.5):
        super(IPNNFrn, self).__init__(field_dims, embed_dim, FRN)
        num_fields = len(field_dims)
        self.pnn = InnerProductNetwork(num_fields)
        self.embed_output_dim = num_fields * embed_dim
        self.inter_size = num_fields * (num_fields - 1) // 2
        self.mlp = MultiLayerPerceptron(self.inter_size + self.input_dim, mlp_layers, dropout=dropout)

    def forward(self, x):
        x_emb = self.embedding(x)
        x_emb, weight = self.frn(x_emb)
        cross_ipnn = self.pnn(x_emb)

        x = torch.cat([x_emb.view(-1, self.embed_output_dim), cross_ipnn], dim=1)
        pred_y = self.mlp(x)
        return pred_y


class OPNN(nn.Module):
    def __init__(self, field_dims, embed_dim, mlp_layers=(400, 400, 400), dropout=0.5, kernel_type="vec"):
        super(OPNN, self).__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        num_fields = len(field_dims)
        if kernel_type == "original":
            self.pnn = OuterProductNetwork2(num_fields, embed_dim)
        else:
            self.pnn = OuterProductNetwork(num_fields, embed_dim, kernel_type)

        self.embed_output_dim = num_fields * embed_dim
        self.inter_size = num_fields * (num_fields - 1) // 2
        self.mlp = MultiLayerPerceptron(self.inter_size + self.embed_output_dim, mlp_layers, dropout)

    def forward(self, x):
        x_emb = self.embedding(x)
        cross_opnn = self.pnn(x_emb)

        x = torch.cat([x_emb.view(-1, self.embed_output_dim), cross_opnn], dim=1)
        pred_y = self.mlp(x)
        return pred_y


class OPNNFrn(BasicFRCTR):
    def __init__(self, field_dims, embed_dim, FRN=None, mlp_layers=(400, 400, 400),
                 dropout=0.5, kernel_type="vec"):
        super(OPNNFrn, self).__init__(field_dims, embed_dim, FRN)
        num_fields = len(field_dims)
        if kernel_type == "original":
            self.pnn = OuterProductNetwork2(num_fields, embed_dim)
        else:
            self.pnn = OuterProductNetwork(num_fields, embed_dim, kernel_type)

        self.embed_output_dim = num_fields * embed_dim
        self.inter_size = num_fields * (num_fields - 1) // 2
        self.mlp = MultiLayerPerceptron(self.inter_size + self.embed_output_dim, mlp_layers, dropout)

    def forward(self, x):
        x_emb = self.embedding(x)
        x_emb, weight = self.frn(x_emb)
        cross_opnn = self.pnn(x_emb)

        x = torch.cat([x_emb.view(-1, self.embed_output_dim), cross_opnn], dim=1)
        pred_y = self.mlp(x)
        return pred_y
