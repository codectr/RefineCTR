# -*- coding: UTF-8 -*-
"""
@project:RefineCTR
"""
import torch.nn as nn

from FRCTR.common import CompressedInteractionNetwork, FeaturesEmbedding, \
    FeaturesLinear, MultiLayerPerceptron, BasicFRCTR


class CIN(nn.Module):
    def __init__(self, field_dims, embed_dim, cross_layer_sizes=(100, 100), split_half=False):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.cin = CompressedInteractionNetwork(len(field_dims), cross_layer_sizes, split_half)

    def forward(self, x):
        """
         param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x_emb = self.embedding(x)
        pred_y = self.cin(x_emb)
        return pred_y


class xDeepFM(nn.Module):
    def __init__(self, field_dims, embed_dim, mlp_dims=(400, 400, 400),
                 dropout=0.5, cross_layer_sizes=(100, 100), split_half=True):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.cin = CompressedInteractionNetwork(len(field_dims), cross_layer_sizes, split_half)
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)
        self.linear = FeaturesLinear(field_dims)

    def forward(self, x):
        x_emb = self.embedding(x)

        cin_term = self.cin(x_emb)
        mlp_term = self.mlp(x_emb.view(-1, self.embed_output_dim))

        pred_y = self.linear(x) + cin_term + mlp_term
        return pred_y


class CINFrn(BasicFRCTR):
    def __init__(self, field_dims, embed_dim, FRN=None,
                 cross_layer_sizes=(100, 100), split_half=False):
        super().__init__(field_dims, embed_dim, FRN)
        self.cin = CompressedInteractionNetwork(len(field_dims), cross_layer_sizes, split_half)

    def forward(self, x):
        x_emb = self.embedding(x)
        x_emb, _ = self.frn(x_emb)
        pred_y = self.cin(x_emb)
        return pred_y


class xDeepFMFrn(BasicFRCTR):
    def __init__(self, field_dims, embed_dim, FRN=None, mlp_dims=(400, 400, 400), dropout=0.5,
                 cross_layer_sizes=(100, 100), split_half=True):
        super().__init__(field_dims, embed_dim, FRN)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.cin = CompressedInteractionNetwork(len(field_dims), cross_layer_sizes, split_half)
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)
        self.linear = FeaturesLinear(field_dims)
        self.frn = FRN

    def forward(self, x):
        x_emb = self.embedding(x)
        x_emb, _ = self.frn(x_emb)
        cin_term = self.cin(x_emb)
        mlp_term = self.mlp(x_emb.view(-1, self.embed_output_dim))

        pred_y = self.linear(x) + cin_term + mlp_term
        return pred_y


class xDeepFMFrnP(nn.Module):
    def __init__(self, field_dims, embed_dim, FRN1=None, FRN2=None, mlp_dims=(400, 400, 400), dropout=0.5,
                 cross_layer_sizes=(100, 100), split_half=True):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        if not FRN1 or not FRN2:
            raise ValueError("Feature Refinement Network is None")
        self.frn1 = FRN1
        self.frn2 = FRN2
        self.embed_output_dim = len(field_dims) * embed_dim
        self.cin = CompressedInteractionNetwork(len(field_dims), cross_layer_sizes, split_half)
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)
        self.linear = FeaturesLinear(field_dims)

    def forward(self, x):
        x_emb = self.embedding(x)
        x_emb1, _ = self.frn1(x_emb)
        x_emb2, _ = self.frn2(x_emb)
        cin_term = self.cin(x_emb1)
        mlp_term = self.mlp(x_emb2.view(-1, self.embed_output_dim))

        pred_y = self.linear(x) + cin_term + mlp_term
        return pred_y
