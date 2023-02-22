# -*- coding: UTF-8 -*-
"""
@project:RefineCTR
"""

import torch
import torch.nn.functional as F

from FRCTR.common import FeaturesEmbedding, FeaturesLinear, MultiLayerPerceptron


class AutoIntPlus(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, atten_embed_dim=64, num_heads=2,
                 num_layers=3, mlp_dims=(400, 400, 400), dropouts=(0.5, 0.5), has_residual=True):
        super().__init__()
        self.num_fields = len(field_dims)
        self.linear = FeaturesLinear(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.atten_embedding = torch.nn.Linear(embed_dim, atten_embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.atten_output_dim = len(field_dims) * atten_embed_dim
        self.has_residual = has_residual
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropouts[1])

        self.self_attns = torch.nn.ModuleList([
            torch.nn.MultiheadAttention(atten_embed_dim, num_heads, dropout=dropouts[0]) for _ in range(num_layers)
        ])
        self.attn_fc = torch.nn.Linear(self.atten_output_dim, 1)
        if self.has_residual:
            self.V_res_embedding = torch.nn.Linear(embed_dim, atten_embed_dim)

    def forward(self, x):
        x_emb = self.embedding(x)
        atten_x = self.atten_embedding(x_emb)

        cross_term = atten_x.transpose(0, 1)

        for self_attn in self.self_attns:
            cross_term, _ = self_attn(cross_term, cross_term, cross_term)

        cross_term = cross_term.transpose(0, 1)
        if self.has_residual:
            V_res = self.V_res_embedding(x_emb)
            cross_term += V_res

        cross_term = F.relu(cross_term).contiguous().view(-1, self.atten_output_dim)

        pred_y = self.linear(x) + self.attn_fc(cross_term) + self.mlp(x_emb.view(-1, self.embed_output_dim))
        return pred_y

class AutoIntPlusFrn(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, FRN=None, atten_embed_dim=64, num_heads=2,
                 num_layers=3, mlp_dims=(400, 400, 400), dropouts=(0.5, 0.5), has_residual=True):
        super().__init__()
        self.num_fields = len(field_dims)
        self.linear = FeaturesLinear(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.atten_embedding = torch.nn.Linear(embed_dim, atten_embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.atten_output_dim = len(field_dims) * atten_embed_dim
        self.has_residual = has_residual
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropouts[1])

        self.self_attns = torch.nn.ModuleList([
            torch.nn.MultiheadAttention(atten_embed_dim, num_heads, dropout=dropouts[0]) for _ in range(num_layers)
        ])
        self.attn_fc = torch.nn.Linear(self.atten_output_dim, 1)
        if self.has_residual:
            self.V_res_embedding = torch.nn.Linear(embed_dim, atten_embed_dim)
        self.frn = FRN
    def forward(self, x):
        x_emb = self.embedding(x)
        x_emb, _ = self.frn(x_emb)
        atten_x = self.atten_embedding(x_emb)

        cross_term = atten_x.transpose(0, 1)

        for self_attn in self.self_attns:
            cross_term, _ = self_attn(cross_term, cross_term, cross_term)

        cross_term = cross_term.transpose(0, 1)
        if self.has_residual:
            V_res = self.V_res_embedding(x_emb)
            cross_term += V_res

        cross_term = F.relu(cross_term).contiguous().view(-1, self.atten_output_dim)

        pred_y = self.linear(x) + self.attn_fc(cross_term) + self.mlp(x_emb.view(-1, self.embed_output_dim))
        return pred_y

class AutoIntPlusFrnP(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, FRN1=None, FRN2=None, atten_embed_dim=64, num_heads=2,
                 num_layers=3, mlp_dims=(400, 400, 400), dropouts=(0.5, 0.5), has_residual=True):
        super().__init__()
        self.num_fields = len(field_dims)
        self.linear = FeaturesLinear(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.atten_embedding = torch.nn.Linear(embed_dim, atten_embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.atten_output_dim = len(field_dims) * atten_embed_dim
        self.has_residual = has_residual
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropouts[1])

        self.self_attns = torch.nn.ModuleList([
            torch.nn.MultiheadAttention(atten_embed_dim, num_heads, dropout=dropouts[0]) for _ in range(num_layers)
        ])
        self.attn_fc = torch.nn.Linear(self.atten_output_dim, 1)
        if self.has_residual:
            self.V_res_embedding = torch.nn.Linear(embed_dim, atten_embed_dim)
        self.frn1 = FRN1
        self.frn2 = FRN2

    def forward(self, x):
        x_emb = self.embedding(x)
        x_emb1, _ = self.frn1(x_emb)
        x_emb2, _ = self.frn2(x_emb)
        x_mlp = self.mlp(x_emb2.view(-1, self.embed_output_dim))

        atten_x = self.atten_embedding(x_emb1)
        cross_term = atten_x.transpose(0, 1)

        for self_attn in self.self_attns:
            cross_term, _ = self_attn(cross_term, cross_term, cross_term)

        cross_term = cross_term.transpose(0, 1)
        if self.has_residual:
            V_res = self.V_res_embedding(x_emb)
            cross_term += V_res

        cross_term = F.relu(cross_term).contiguous().view(-1, self.atten_output_dim)

        pred_y = self.linear(x) + self.attn_fc(cross_term) + x_mlp
        return pred_y


class AutoInt(torch.nn.Module):

    def __init__(self, field_dims, embed_dim, atten_embed_dim=32, num_heads=4,
                 num_layers=3, dropouts=(0.5, 0.5), has_residual=True):
        super().__init__()
        self.num_fields = len(field_dims)
        self.linear = FeaturesLinear(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.atten_embedding = torch.nn.Linear(embed_dim, atten_embed_dim)
        self.atten_output_dim = len(field_dims) * atten_embed_dim
        self.has_residual = has_residual

        self.self_attns = torch.nn.ModuleList([
            torch.nn.MultiheadAttention(atten_embed_dim, num_heads, dropout=dropouts[0]) for _ in range(num_layers)
        ])
        self.attn_fc = torch.nn.Linear(self.atten_output_dim, 1)
        if self.has_residual:
            self.V_res_embedding = torch.nn.Linear(embed_dim, atten_embed_dim)

    def forward(self, x):
        x_emb = self.embedding(x)
        atten_x = self.atten_embedding(x_emb)

        cross_term = atten_x.transpose(0, 1)

        for self_attn in self.self_attns:
            cross_term, _ = self_attn(cross_term, cross_term, cross_term)

        cross_term = cross_term.transpose(0, 1)
        if self.has_residual:
            V_res = self.V_res_embedding(x_emb)
            cross_term += V_res

        cross_term = F.relu(cross_term).contiguous().view(-1, self.atten_output_dim)

        pred_y = self.attn_fc(cross_term) + self.linear(x)
        return pred_y

class AutoIntFrn(torch.nn.Module):

    def __init__(self, field_dims, embed_dim, FRN=None, atten_embed_dim=32, num_heads=4,
                 num_layers=3, dropouts=(0.5, 0.5), has_residual=True):
        super().__init__()
        self.num_fields = len(field_dims)
        self.linear = FeaturesLinear(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.atten_embedding = torch.nn.Linear(embed_dim, atten_embed_dim)
        self.atten_output_dim = len(field_dims) * atten_embed_dim
        self.has_residual = has_residual

        self.self_attns = torch.nn.ModuleList([
            torch.nn.MultiheadAttention(atten_embed_dim, num_heads, dropout=dropouts[0]) for _ in range(num_layers)
        ])
        self.attn_fc = torch.nn.Linear(self.atten_output_dim, 1)
        if self.has_residual:
            self.V_res_embedding = torch.nn.Linear(embed_dim, atten_embed_dim)

        self.frn = FRN

    def forward(self, x):
        x_emb = self.embedding(x)
        x_emb, _ = self.frn(x_emb)
        atten_x = self.atten_embedding(x_emb)

        cross_term = atten_x.transpose(0, 1)

        for self_attn in self.self_attns:
            cross_term, _ = self_attn(cross_term, cross_term, cross_term)

        cross_term = cross_term.transpose(0, 1)
        if self.has_residual:
            V_res = self.V_res_embedding(x_emb)
            cross_term += V_res

        cross_term = F.relu(cross_term).contiguous().view(-1, self.atten_output_dim)

        pred_y = self.attn_fc(cross_term) + self.linear(x)
        return pred_y
