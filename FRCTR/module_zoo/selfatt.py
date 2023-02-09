# -*- coding: UTF-8 -*-
"""
@project:RefineCTR
"""
import torch
from torch import nn


class InterCTRLayer(nn.Module):
    def __init__(self, embed_dim=16, att_size=32, num_heads=8, out_dim=32, use_res=False):
        super(InterCTRLayer, self).__init__()
        self.use_res = use_res
        self.dim_per_head = att_size
        self.num_heads = num_heads

        self.linear_k = nn.Linear(embed_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(embed_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(embed_dim, self.dim_per_head * num_heads)

        self.outputw = torch.nn.Linear(self.dim_per_head * num_heads, out_dim, bias=False)
        if self.use_res:
            # self.linear_residual = nn.Linear(model_dim, self.dim_per_head * num_heads)
            self.linear_residual = nn.Linear(embed_dim, out_dim)
        nn.init.xavier_uniform_(self.linear_q.weight)
        nn.init.xavier_uniform_(self.linear_k.weight)
        nn.init.xavier_uniform_(self.linear_v.weight)
        nn.init.xavier_uniform_(self.outputw.weight)

    def _dot_product_attention(self, q, k, v):
        attention = torch.bmm(q, k.transpose(1, 2))

        attention = torch.softmax(attention, dim=2)

        attention = torch.dropout(attention, p=0.0, train=self.training)
        context = torch.bmm(attention, v)
        return context, attention

    def forward(self, query):
        batch_size = query.size(0)
        key = self.linear_k(query)
        value = self.linear_v(query)
        query = self.linear_q(query)

        key = key.view(batch_size * self.num_heads, -1, self.dim_per_head)
        value = value.view(batch_size * self.num_heads, -1, self.dim_per_head)
        query = query.view(batch_size * self.num_heads, -1, self.dim_per_head)

        context, attention = self._dot_product_attention(query, key, value)  # [B*16, 10, 256]
        context = context.view(batch_size, -1, self.dim_per_head * self.num_heads)  # [B, 10, 256*16]
        context = torch.relu(self.outputw(context))  # B, F, out_dim
        return context, attention
