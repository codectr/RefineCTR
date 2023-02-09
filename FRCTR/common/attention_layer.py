# -*- coding: UTF-8 -*-
"""
@project: RefineCTR
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AttentionLayer(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.dim = dim
        self.q_layer = nn.Linear(dim, dim, bias=False)
        self.k_layer = nn.Linear(dim, dim, bias=False)
        self.v_layer = nn.Linear(dim, dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        Q = self.q_layer(x)
        K = self.k_layer(x)
        V = self.v_layer(x)
        a = torch.sum(torch.mul(Q, K), -1) / torch.sqrt(torch.tensor(self.dim))
        a = self.softmax(a)
        outputs = torch.sum(torch.mul(torch.unsqueeze(a, -1), V), dim=1)
        return outputs


class FieldAttentionModule(nn.Module):
    def __init__(self, embed_dim):
        super(FieldAttentionModule, self).__init__()
        self.trans_Q = nn.Linear(embed_dim, embed_dim)
        self.trans_K = nn.Linear(embed_dim, embed_dim)
        self.trans_V = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, scale=None, mask=None):
        Q = self.trans_Q(x)
        K = self.trans_K(x)
        V = self.trans_V(x)

        attention = torch.matmul(Q, K.permute(0, 2, 1))
        if scale:
            attention = attention * scale
        if mask:
            attention = attention.masked_fill_(mask == 0, -1e9)
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)

        return context


class Attention(nn.Module):
    def __init__(self, method='dot', hidden_size=None):
        super(Attention, self).__init__()
        self.method = method
        if self.method != 'dot':
            self.hidden_size = hidden_size
            if self.method == 'general':
                self.W = nn.Linear(hidden_size, hidden_size)
            elif self.method == 'concat':
                self.W = nn.Linear(self.hidden_size * 2, hidden_size)
                self.v = nn.Parameter(torch.rand(1, hidden_size))
                nn.init.xavier_normal_(self.v.data)

    def forward(self, query, key, value, mask=None, dropout=0):
        if self.method == 'general':
            scores = self.general(query, key)
        elif self.method == 'concat':
            scores = self.concat(query, key)
        else:
            scores = self.dot(query, key)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if not dropout:
            p_attn = F.dropout(p_attn, dropout)

        return torch.matmul(p_attn, value), p_attn

    def dot(self, query, key):
        scores = torch.matmul(query, key.transpose(-2, -1))
        return scores

    def general(self, query, key):
        scores = torch.matmul(self.W(query), key.transpose(-2, -1))
        return scores

    def concat(self, query, key):
        scores = torch.cat((query.expand(-1, key.size(1), -1), key), dim=2)
        scores = self.W(scores)
        scores = F.tanh(scores)
        scores = torch.matmul(scores, self.v.t()).transpose(-2, -1)
        return scores


class GeneralAttention(nn.Module):
    def __init__(self, embed_dim, conv_size=0):
        super(GeneralAttention, self).__init__()
        if conv_size == 0:
            conv_size = embed_dim
        # self.attention = torch.nn.Linear(embed_dim, embed_dim)
        self.attention = torch.nn.Linear(embed_dim, conv_size)
        self.projection = torch.nn.Linear(conv_size, 1)
        # self.projection = torch.nn.Linear(embed_dim, 1)

    def forward(self, key, dim=1):
        attn_scores = F.relu(self.attention(key))
        attn_scores = F.softmax(self.projection(attn_scores), dim=dim)
        attn_output = torch.sum(attn_scores * key, dim=dim)  # B,e
        return attn_output, attn_scores


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, dropout=.1):
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):

        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)
        # scale = (key.size(-1) // num_heads) ** -0.5
        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att = self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out

class ScaleDotProductAttention(nn.Module):

    def forward(self, q, k, v, scale=None, attn_mask=None):
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale
        if attn_mask:
            attention = attention.masked_fill(attn_mask, -np.inf)
        attention = torch.softmax(attention, dim=2)

        attention = torch.dropout(attention, p=0.0, train=self.training)
        context = torch.bmm(attention, v)
        return context, attention


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim=20, dk=32, num_heads=16, out_dim=32, use_res = True):
        super(MultiHeadAttention, self).__init__()
        self.use_res = use_res
        self.dim_per_head = dk
        self.num_heads = num_heads

        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.outputw = torch.nn.Linear(self.dim_per_head * num_heads, out_dim)
        if self.use_res:
            # self.linear_residual = nn.Linear(model_dim, self.dim_per_head * num_heads)
            self.linear_residual = nn.Linear(model_dim, out_dim)
        # self.dot_product_attention = DotAttention()
        # self.linear_final = nn.Linear(model_dim, model_dim)
        # self.linear_residual = nn.Linear(model_dim, self.dim_per_head * num_heads)
        # self.layer_norm = nn.LayerNorm(model_dim)  # LayerNorm 归一化。

    def _dot_product_attention(self, q, k, v, scale=None, attn_mask=None):
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale
        if attn_mask:
            attention = attention.masked_fill(attn_mask, -np.inf)
        # score = softmax(QK^T / (d_k ** 0.5))
        attention = torch.softmax(attention, dim=2)

        attention = torch.dropout(attention, p=0.0, train=self.training)
        #  out = score * V
        context = torch.bmm(attention, v)
        return context, attention

    def forward(self, query, key, value, attn_mask=None):
        batch_size = key.size(0)

        key = self.linear_k(key)  # K = UWk [B, 10, 256*16]
        value = self.linear_v(value)  # Q = UWv [B, 10, 256*16]
        query = self.linear_q(query)  # V = UWq [B, 10, 256*16]

        # [B, 10, 256*16] =》 [B*16, 10, 256]
        key = key.view(batch_size * self.num_heads, -1, self.dim_per_head)
        value = value.view(batch_size * self.num_heads, -1, self.dim_per_head)
        query = query.view(batch_size * self.num_heads, -1, self.dim_per_head)

        if attn_mask:
            attn_mask = attn_mask.unsqueeze(1).repeat(self.num_heads*batch_size, query.size(1), 1)

        scale = (key.size(-1) // self.num_heads) ** -0.5
        # QK^T/(dk**0.5) * V
        context, attention = self._dot_product_attention(query, key, value, scale, attn_mask) # [B*16, 10, 256]

        context = context.view(batch_size, -1, self.dim_per_head * self.num_heads)  # [B, 10, 256*16]
        context = self.outputw(context) # B, F, out_dim

        if self.use_res:
            context += self.linear_residual(query) # B, F, out_dim

        return context, attention

class MultiHeadAttention2(nn.Module):

    def __init__(self, query_dim, key_dim, num_units, num_heads):

        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)

    def forward(self, query, key, mask=None):
        querys = self.W_query(query)  # [N, T_q, num_units]
        keys = self.W_key(key)  # [N, T_k, num_units]
        values = self.W_value(key)

        split_size = self.num_units // self.num_heads
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)  # [h, N, T_q, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]

        scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        scores = scores / (self.key_dim ** 0.5)

        if mask:
            mask = mask.unsqueeze(1).unsqueeze(0).repeat(self.num_heads,1,querys.shape[2],1)
            scores = scores.masked_fill(mask, -np.inf)
        scores = F.softmax(scores, dim=3)

        out = torch.matmul(scores, values)  # [h, N, T_q, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]

        return out,scores
