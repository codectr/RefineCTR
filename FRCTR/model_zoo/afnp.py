# -*- coding: UTF-8 -*-
"""
@project:RefineCTR
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from FRCTR.common import FeaturesEmbedding, MultiLayerPerceptron, BasicFRCTR, FeaturesLinear


class LNN(torch.nn.Module):
    """
    A pytorch implementation of LNN layer
    Input shape
        - A 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
    Output shape
        - 2D tensor with shape:``(batch_size,LNN_dim*embedding_size)``.
    Arguments
        - **in_features** : Embedding of feature.
        - **num_fields**: int.The field size of feature.
        - **LNN_dim**: int.The number of Logarithmic neuron.
        - **bias**: bool.Whether or not use bias in LNN.
    """

    def __init__(self, num_fields, embed_dim, LNN_dim, bias=False):
        super(LNN, self).__init__()
        self.num_fields = num_fields
        self.embed_dim = embed_dim
        self.LNN_dim = LNN_dim
        self.lnn_output_dim = LNN_dim * embed_dim

        self.weight = torch.nn.Parameter(torch.Tensor(LNN_dim, num_fields))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(LNN_dim, embed_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        """
            param x: Long tensor of size ``(batch_size, num_fields, embedding_size)``
        """
        # Computes the element-wise absolute value of the given input tensor.
        embed_x_abs = torch.abs(x)
        embed_x_afn = torch.add(embed_x_abs, 1e-7)
        # Logarithmic Transformation
        # torch.log1p
        embed_x_log = torch.log1p(embed_x_afn)
        lnn_out = torch.matmul(self.weight, embed_x_log)
        if self.bias is not None:
            lnn_out += self.bias

        # torch.expm1
        lnn_exp = torch.expm1(lnn_out)
        output = F.relu(lnn_exp).contiguous().view(-1, self.lnn_output_dim)
        return output


class AFN(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, LNN_dim=10, mlp_dims=(400, 400, 400), dropouts=(0.5, 0.5)):
        super().__init__()
        self.linear = FeaturesLinear(field_dims)
        self.num_fields = len(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.LNN_dim = LNN_dim
        self.LNN_output_dim = self.LNN_dim * embed_dim
        self.LNN = LNN(self.num_fields, embed_dim, LNN_dim)

        self.mlp = MultiLayerPerceptron(self.LNN_output_dim, mlp_dims, dropouts[0])

    def forward(self, x):

        x_emb = self.embedding(x)

        lnn_out = self.LNN(x_emb)

        pred_y = self.mlp(lnn_out) + self.linear(x)
        return pred_y


class AFNPlus(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, LNN_dim=10, mlp_dims=(400, 400, 400),
                 mlp_dims2=(400, 400, 400), dropouts=(0.5, 0.5)):
        super().__init__()
        self.num_fields = len(field_dims)
        self.linear = FeaturesLinear(field_dims)  # Linear
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)  # Embedding

        self.LNN_dim = LNN_dim
        self.LNN_output_dim = self.LNN_dim * embed_dim
        self.LNN = LNN(self.num_fields, embed_dim, LNN_dim)

        self.mlp = MultiLayerPerceptron(self.LNN_output_dim, mlp_dims, dropouts[0])

        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp2 = MultiLayerPerceptron(self.embed_output_dim, mlp_dims2, dropouts[1])

        self.lr = torch.nn.Linear(2, 1, bias=True)

    def forward(self, x):
        """
         param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x_emb = self.embedding(x)

        lnn_out = self.LNN(x_emb)
        x_dnn = self.mlp2(x_emb.view(-1, self.embed_output_dim))
        x_lnn = self.mlp(lnn_out)
        pred_y = self.linear(x) + x_lnn + x_dnn
        return pred_y


class AFNFrn(BasicFRCTR):
    def __init__(self, field_dims, embed_dim, FRN=None, LNN_dim=16, mlp_dims=(400, 400, 400), dropouts=(0.5, 0.5)):
        super().__init__(field_dims, embed_dim, FRN)
        self.linear = FeaturesLinear(field_dims)
        self.num_fields = len(field_dims)
        self.LNN_dim = LNN_dim
        self.LNN_output_dim = self.LNN_dim * embed_dim
        self.LNN = LNN(self.num_fields, embed_dim, LNN_dim)
        self.mlp = MultiLayerPerceptron(self.LNN_output_dim, mlp_dims, dropouts[0], output_layer=True)

    def forward(self, x):
        """
         param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x_emb = self.embedding(x)
        x_emb, _ = self.frn(x_emb)

        lnn_out = self.LNN(x_emb)

        pred_y = self.mlp(lnn_out) + self.linear(x)
        return pred_y


class AFNPlusFrn(BasicFRCTR):
    def __init__(self, field_dims, embed_dim, FRN=None, LNN_dim=10, mlp_dims=(400, 400, 400),
                 mlp_dims2=(400, 400, 400), dropouts=(0.5, 0.5)):
        super().__init__(field_dims, embed_dim, FRN)
        self.linear = FeaturesLinear(field_dims)
        self.num_fields = len(field_dims)

        self.LNN_dim = LNN_dim
        self.LNN_output_dim = self.LNN_dim * embed_dim
        self.LNN = LNN(self.num_fields, embed_dim, LNN_dim)

        self.mlp = MultiLayerPerceptron(self.LNN_output_dim, mlp_dims, dropouts[0])

        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp2 = MultiLayerPerceptron(self.embed_output_dim, mlp_dims2, dropouts[1])

        self.lr = torch.nn.Linear(2, 1, bias=True)

    def forward(self, x):
        """
         param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x_emb = self.embedding(x)
        x_emb, _ = self.frn1(x_emb)

        lnn_out = self.LNN(x_emb)
        x_dnn = self.mlp2(x_emb.view(-1, self.embed_output_dim))
        x_lnn = self.mlp(lnn_out)
        pred_y = self.linear(x) + x_lnn + x_dnn
        return pred_y


class AFNPlusFrnP(nn.Module):
    def __init__(self, field_dims, embed_dim, FRN1=None, FRN2=None, LNN_dim=10, mlp_dims=(400, 400, 400),
                 mlp_dims2=(400, 400, 400), dropouts=(0.5, 0.5)):
        super().__init__()
        self.num_fields = len(field_dims)
        self.linear = FeaturesLinear(field_dims)  # Linear
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)  # Embedding

        if not FRN1 or not FRN2:
            raise ValueError("Feature Refinement Network is None")
        self.frn1 = FRN1
        self.frn2 = FRN2

        self.LNN_dim = LNN_dim
        self.LNN_output_dim = self.LNN_dim * embed_dim
        self.LNN = LNN(self.num_fields, embed_dim, LNN_dim)

        self.mlp = MultiLayerPerceptron(self.LNN_output_dim, mlp_dims, dropouts[0])

        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp2 = MultiLayerPerceptron(self.embed_output_dim, mlp_dims2, dropouts[1])

        self.lr = torch.nn.Linear(2, 1, bias=True)

    def forward(self, x):
        x_emb = self.embedding(x)
        x_emb1, weight1 = self.frn1(x_emb)
        x_emb2, weight2 = self.frn2(x_emb)

        lnn_out = self.LNN(x_emb1)
        x_lnn = self.mlp(lnn_out)

        x_dnn = self.mlp2(x_emb2.reshape(-1, self.embed_output_dim))
        pred_y = self.linear(x) + x_lnn + x_dnn
        # pred_y = self.lr(torch.cat([x_lnn, x_dnn], dim=1))
        return pred_y
