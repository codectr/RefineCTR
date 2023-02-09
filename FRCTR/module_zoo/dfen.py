# -*- coding: UTF-8 -*-
"""
@project:RefineCTR
"""

import torch
from torch import nn


class DualFENLayer(nn.Module):
    def __init__(self, field_length, embed_dim, embed_dims=[256, 256, 256], att_size=64, num_heads=8):
        super(DualFENLayer, self).__init__()
        input_dim = field_length * embed_dim  # 10*256
        self.mlp = MultiLayerPerceptron(input_dim, embed_dims, dropout=0.5, output_layer=False)

        self.multihead = MultiHeadAttentionL(model_dim=embed_dim, dk=att_size, num_heads=num_heads)
        self.trans_vec_size = att_size * num_heads * field_length
        self.trans_vec = nn.Linear(self.trans_vec_size, field_length, bias=False)
        self.trans_bit = nn.Linear(embed_dims[-1], field_length, bias=False)

    def forward(self, x_emb):
        # (1) concat
        x_con = x_emb.view(x_emb.size(0), -1)  # [B, ?]

        # （2）bit-level difm does not apply softmax or sigmoid
        m_bit = self.mlp(x_con)

        # （3）vector-level multi-head
        x_att2 = self.multihead(x_emb, x_emb, x_emb)
        m_vec = self.trans_vec(x_att2.view(-1, self.trans_vec_size))
        m_bit = self.trans_bit(m_bit)

        x_att = m_bit + m_vec
        x_emb = x_emb * x_att.unsqueeze(2)
        return x_emb, x_att


class MultiHeadAttentionL(nn.Module):
    def __init__(self, model_dim=256, dk=32, num_heads=16):
        super(MultiHeadAttentionL, self).__init__()

        self.dim_per_head = dk  # dk dv
        self.num_heads = num_heads

        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.linear_residual = nn.Linear(model_dim, self.dim_per_head * num_heads)

        # self.layer_norm = nn.LayerNorm(model_dim)  # LayerNorm

    def _dot_product_attention(self, q, k, v, scale=None):
        attention = torch.bmm(q, k.transpose(1, 2)) * scale
        attention = torch.softmax(attention, dim=2)
        attention = torch.dropout(attention, p=0.0, train=self.training)
        context = torch.bmm(attention, v)
        return context, attention

    def forward(self, key0, value0, query0, attn_mask=None):
        batch_size = key0.size(0)

        key = self.linear_k(key0)
        value = self.linear_v(value0)
        query = self.linear_q(query0)

        key = key.view(batch_size * self.num_heads, -1, self.dim_per_head)
        value = value.view(batch_size * self.num_heads, -1, self.dim_per_head)
        query = query.view(batch_size * self.num_heads, -1, self.dim_per_head)

        scale = (key.size(-1) // self.num_heads) ** -0.5
        context, attention = self._dot_product_attention(query, key, value, scale)
        context = context.view(batch_size, -1, self.dim_per_head * self.num_heads)

        residual = self.linear_residual(query0)
        residual = residual.view(batch_size, -1, self.dim_per_head * self.num_heads)  # [B, 10, 256*h]

        output = torch.relu(residual + context)  # [B, 10, 256]
        return output


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, embed_dims, dropout=0.5, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
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
