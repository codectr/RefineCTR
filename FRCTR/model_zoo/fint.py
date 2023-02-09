# -*- coding: UTF-8 -*-
"""
@project:RefineCTR
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from FRCTR.common import FeaturesLinear, FeaturesEmbedding, FactorizationMachine, \
    MultiLayerPerceptron, BasicFRCTR

class FintLayer(nn.Module):
    def forward(self, x_vl, x_wl, x_ul, x_embed):
        x_vl = x_vl * (torch.matmul(x_wl, x_embed)) + x_ul * x_vl
        return x_vl


class FINT(nn.Module):
    """
    1、Embedding layer
    2、Field aware interaction layer。
    3、DNN layer for prediction
    """

    def __init__(self, field_dims, embed_dim, mlp_layers=(400, 400, 400), num_deep=3, dropout=0.5):
        super(FINT, self).__init__()

        self.linear = FeaturesLinear(field_dims)

        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        num_fields = len(field_dims)
        self.num_deep = num_deep
        self.fints_param = nn.ParameterList(
            [nn.Parameter(torch.randn(num_fields, num_fields)) for _ in range(num_deep)])
        self.Ul = nn.ParameterList([nn.Parameter(torch.ones(1, num_fields, 1)) for _ in range(num_deep)])

        self.embed_output_size = num_fields * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_size, embed_dims=mlp_layers, dropout=0.5)

    def forward(self, x):
        x_emb = self.embedding(x)
        x_fint = x_emb
        for i in range(self.num_deep):
            x_fint = x_fint * torch.matmul(self.fints_param[i], x_emb) + self.Ul[i] * x_fint

        pred_y = self.mlp(x_fint.view(x_fint.size(0), -1))
        return pred_y

class FINTFrn(BasicFRCTR):
    """
    1、Embedding layer
    2、Field aware interaction layer。
    3、DNN layer for prediction
    """

    def __init__(self, field_dims, embed_dim, FRN=None, mlp_layers=(400, 400, 400), num_deep=3, dropout=0.5):
        super(FINTFrn, self).__init__(field_dims, embed_dim, FRN)

        self.linear = FeaturesLinear(field_dims)
        num_fields = len(field_dims)
        self.num_deep = num_deep
        self.fints_param = nn.ParameterList(
            [nn.Parameter(torch.randn(num_fields, num_fields)) for _ in range(num_deep)])
        self.Ul = nn.ParameterList([nn.Parameter(torch.ones(1, num_fields, 1)) for _ in range(num_deep)])

        self.embed_output_size = num_fields * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_size, embed_dims=mlp_layers, dropout=0.5)

    def forward(self, x):
        x_emb = self.embedding(x)
        x_emb, weight = self.frn(x_emb)
        x_fint = x_emb
        for i in range(self.num_deep):
            x_fint = x_fint * torch.matmul(self.fints_param[i], x_emb) + self.Ul[i] * x_fint

        pred_y = self.mlp(x_fint.view(x_fint.size(0), -1))
        return pred_y