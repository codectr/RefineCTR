# -*- coding: UTF-8 -*-
"""
@project:RefineCTR
"""
import torch
import torch.nn.functional as F
from torch import nn


class FRNetLayer(nn.Module):
    def __init__(self, field_length, embed_dim, weight_type="bit",
                 num_layers=1, att_size=10, mlp_layer=256):
        super(FRNetLayer, self).__init__()
        self.IEU_G = IEU(field_length, embed_dim, weight_type="bit",
                         bit_layers=num_layers, att_size=att_size, mlp_layer=mlp_layer)

        # bit-level or vector-level weights.
        self.IEU_W = IEU(field_length, embed_dim, weight_type=weight_type,
                         bit_layers=num_layers, att_size=att_size, mlp_layer=mlp_layer)

    def forward(self, x_embed):
        weight_matrix = torch.sigmoid(self.IEU_W(x_embed))
        com_feature = self.IEU_G(x_embed)
        # CSGate
        x_out = x_embed * weight_matrix + com_feature * (torch.tensor(1.0) - weight_matrix)
        return x_out, weight_matrix


class IEU(nn.Module):
    def __init__(self, field_length, embed_dim, weight_type="bit",
                 bit_layers=1, att_size=10, mlp_layer=256):
        super(IEU, self).__init__()
        self.input_dim = field_length * embed_dim
        self.weight_type = weight_type

        # Self-attention unit, which is used to capture cross-feature relationships.
        self.vector_info = SelfAttentionIEU(embed_dim=embed_dim, att_size=att_size)

        #  contextual information extractor(CIE), FRNet adopt MLP to encode contextual information.
        mlp_layers = [mlp_layer for _ in range(bit_layers)]
        self.mlps = MultiLayerPerceptronPrelu(self.input_dim, embed_dims=mlp_layers,
                                              output_layer=False)
        self.bit_projection = nn.Linear(mlp_layer, embed_dim)
        # self.activation = nn.ReLU()
        self.activation = nn.PReLU()

    def forward(self, x_emb):
        # （1）Self-attetnion unit
        x_vector = self.vector_info(x_emb)  # B,F,E

        # (2) CIE unit
        x_bit = self.mlps(x_emb.view(-1, self.input_dim))
        x_bit = self.bit_projection(x_bit).unsqueeze(1)  # B,1,e
        x_bit = self.activation(x_bit)

        # （3）integration unit
        x_out = x_bit * x_vector

        if self.weight_type == "vector":
            # To compute vector-level importance in IEU_W
            x_out = torch.sum(x_out, dim=2, keepdim=True)
            return x_out  # B,F,1

        return x_out  # B,F,E


class SelfAttentionIEU(nn.Module):
    def __init__(self, embed_dim, att_size=20):
        super(SelfAttentionIEU, self).__init__()
        self.embed_dim = embed_dim
        self.trans_Q = nn.Linear(embed_dim, att_size)
        self.trans_K = nn.Linear(embed_dim, att_size)
        self.trans_V = nn.Linear(embed_dim, att_size)
        self.projection = nn.Linear(att_size, embed_dim)
        # self.scale = embed_dim.size(-1)  ** -0.5

    def forward(self, x, scale=None):
        Q = self.trans_Q(x)
        K = self.trans_K(x)
        V = self.trans_V(x)

        attention = torch.matmul(Q, K.permute(0, 2, 1))  # B,F,F
        attention_score = F.softmax(attention, dim=-1)
        context = torch.matmul(attention_score, V)
        context = self.projection(context)
        return context


class MultiLayerPerceptronPrelu(torch.nn.Module):
    def __init__(self, input_dim, embed_dims, dropout=0.5, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.PReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim

        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)
        self._init_weight_()

    def _init_weight_(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        return self.mlp(x)
