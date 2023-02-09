# -*- coding: UTF-8 -*-
"""
@project:RefineCTR
"""

from torch import nn
from FRCTR.common import BasicFRCTR, FeaturesLinear, FeaturesEmbedding, \
    FactorizationMachine, MultiLayerPerceptron


class DeepFM(nn.Module):
    def __init__(self, field_dims, embed_dim, mlp_layers=(400, 400, 400), dropout=0.5):
        super(DeepFM, self).__init__()
        self.lr = FeaturesLinear(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)

        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, embed_dims=mlp_layers,
                                        dropout=dropout, output_layer=True)

        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, x):
        x_emb = self.embedding(x)
        pred_y = self.lr(x) + self.fm(x_emb) + self.mlp(x_emb.view(x.size(0), -1))
        return pred_y


class DeepFMFrn(BasicFRCTR):
    def __init__(self, field_dims, embed_dim, FRN=None, mlp_layers=(400, 400, 400), dropout=0.5):
        super(DeepFMFrn, self).__init__(field_dims, embed_dim, FRN)
        self.lr = FeaturesLinear(field_dims)

        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, embed_dims=mlp_layers,
                                        dropout=dropout, output_layer=True)

        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, x):
        x_emb = self.embedding(x)
        x_emb, x_weight = self.frn(x_emb)
        pred_y = self.lr(x) + self.fm(x_emb) + self.mlp(x_emb.reshape(x.size(0), -1))
        return pred_y

class DeepFMFrnP(nn.Module):
    """
        DeepFM with two separate feature refinement modules.
    """
    def __init__(self, field_dims, embed_dim, FRN1=None, FRN2=None, mlp_layers=(400, 400, 400), dropout=0.5):
        super(DeepFMFrnP, self).__init__()
        self.lr = FeaturesLinear(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)

        if not FRN1 or not FRN2:
            raise ValueError("Feature Refinement Network is None")
        self.frn1 = FRN1
        self.frn2 = FRN2

        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, embed_dims=mlp_layers,
                                        dropout=dropout, output_layer=True)

        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, x):
        x_emb = self.embedding(x)
        x_emb1, x_weight1 = self.frn1(x_emb)
        x_emb2, x_weight2 = self.frn2(x_emb)
        pred_y = self.lr(x) + self.fm(x_emb1) + self.mlp(x_emb2.reshape(x.size(0), -1))
        return pred_y