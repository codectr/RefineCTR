# -*- coding: UTF-8 -*-
"""
@project:RefineCTR
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from FRCTR.common import FeaturesEmbedding, FeaturesLinear, MultiLayerPerceptron

class DeepCrossAttentionalProductNetwork(torch.nn.Module):

    def __init__(self, field_dims, embed_dim, num_heads, num_layers, mlp_dims, dropouts):
        super().__init__()
        num_fields = len(field_dims)
        self.cap = CrossAttentionalProductNetwork(num_fields, embed_dim, num_heads, num_layers, dropouts[0])
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.num_layers = num_layers
        self.embed_output_dim = num_fields * embed_dim
        self.attn_output_dim = num_layers * num_fields * (num_fields - 1) // 2
        self.mlp = MultiLayerPerceptron(self.attn_output_dim + self.embed_output_dim, mlp_dims, dropouts[1])

    def generate_square_subsequent_mask(self, num_fields):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(num_fields, num_fields)) == 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x):
        device = x.device
        attn_mask = self.generate_square_subsequent_mask(x.size(1)).to(device)
        embed_x = self.embedding(x)
        cross_term = self.cap(embed_x, attn_mask)
        y = torch.cat([embed_x.view(-1, self.embed_output_dim), cross_term], dim=1)
        x = self.mlp(y)
        return x.squeeze(1)

class DCAPFrn(torch.nn.Module):
    """
    A pytorch implementation of inner/outer Product Neural Network.
    Reference:
        Y Qu, et al. Product-based Neural Networks for User Response Prediction, 2016.
    """

    def __init__(self, field_dims, embed_dim, num_heads=1, num_layers=3, mlp_dims=(400,400,400), dropouts=(0.5,0.5), FRN=None):
        super().__init__()
        num_fields = len(field_dims)
        self.cap = CrossAttentionalProductNetwork(num_fields, embed_dim, num_heads, num_layers, dropout=dropouts[0])
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.frn = FRN
        self.num_layers = num_layers
        self.embed_output_dim = num_fields * embed_dim
        self.attn_output_dim = num_layers * num_fields * (num_fields - 1) // 2
        self.mlp = MultiLayerPerceptron(self.attn_output_dim + self.embed_output_dim, mlp_dims, dropouts[1])

    def generate_square_subsequent_mask(self, num_fields):
        mask = (torch.triu(torch.ones(num_fields, num_fields)) == 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    def forward(self, x):
        device = x.device
        attn_mask = self.generate_square_subsequent_mask(x.size(1)).to(device)
        x_emb = self.embedding(x)
        x_emb, weight = self.frn(x_emb)
        cross_term = self.cap(x_emb, attn_mask)
        x_cat = torch.cat([x_emb.view(-1, self.embed_output_dim), cross_term], dim=1)
        pred_y = self.mlp(x_cat)
        return pred_y


class CrossAttentionalProductNetwork(torch.nn.Module):

    def __init__(self, num_fields, embed_dim, num_heads, num_layers, dropout, kernel_type='mat'):
        super().__init__()
        self.layers = torch.nn.ModuleList([])
        self.layers.extend(
            [self.build_encoder_layer(num_fields=num_fields, embed_dim=embed_dim, num_heads=num_heads,
                                    dropout=dropout, kernel_type=kernel_type) for _ in range(num_layers)]
        )


    def build_encoder_layer(self, num_fields, embed_dim, num_heads, dropout, kernel_type='mat'):
        return CrossProductNetwork(num_fields=num_fields, embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, kernel_type=kernel_type)

    def forward(self, x, attn_mask=None):
        x0 = x
        output = []
        for layer in self.layers:
            x, y = layer(x, x0, attn_mask)
            output.append(y)
        output = torch.cat(output, dim=1)

        return output


class CrossProductNetwork(torch.nn.Module):

    def __init__(self, num_fields, embed_dim, num_heads, dropout=0.2, kernel_type='mat'):
        super().__init__()
        num_ix = num_fields * (num_fields - 1) // 2
        if kernel_type == 'mat':
            kernel_shape = embed_dim, num_ix, embed_dim
        elif kernel_type == 'vec':
            kernel_shape = num_ix, embed_dim
        elif kernel_type == 'num':
            kernel_shape = num_ix, 1
        else:
            raise ValueError('unknown kernel type: ' + kernel_type)
        self.kernel_type = kernel_type
        self.kernel = torch.nn.Parameter(torch.zeros(kernel_shape))
        self.avg_pool = torch.nn.AdaptiveAvgPool1d(num_fields)
        # self.fc = torch.nn.Linear(embed_dim, 1)
        self.attn = MultiheadAttentionInnerProduct(num_fields, embed_dim, num_heads, dropout)
        torch.nn.init.xavier_uniform_(self.kernel.data)

    def forward(self, x, x0, attn_mask=None):

        bsz, num_fields, embed_dim = x0.size()
        row, col = list(), list()
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                row.append(i), col.append(j)

        x, _ = self.attn(x, x, x, attn_mask)
        p, q = x[:, row], x0[:, col]
        if self.kernel_type == 'mat':
            kp = torch.sum(p.unsqueeze(1) * self.kernel, dim=-1).permute(0, 2, 1)  # (bsz, n(n-1)/2, embed_dim)
            kpq = kp * q

            x = self.avg_pool(kpq.permute(0, 2, 1)).permute(0, 2, 1)  # (bsz, n, embed_dim)

            return x, torch.sum(kpq, dim=-1)
        else:
            return torch.sum(p * q * self.kernel.unsqueeze(0), -1)


class MultiheadAttentionInnerProduct(torch.nn.Module):
    def __init__(self, num_fields, embed_dim, num_heads, dropout):
        super().__init__()
        self.num_fields = num_fields
        self.mask = (torch.triu(torch.ones(num_fields, num_fields), diagonal=1) == 1)
        self.num_cross_terms = num_fields * (num_fields - 1) // 2
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout
        head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim, "head dim is not divisible by embed dim"
        self.head_dim = head_dim
        self.scale = self.head_dim ** -0.5

        self.linear_q = torch.nn.Linear(embed_dim, num_heads * head_dim, bias=True)
        self.linear_k = torch.nn.Linear(embed_dim, num_heads * head_dim, bias=True)
        self.avg_pool = torch.nn.AdaptiveAvgPool1d(num_fields)
        self.output_layer = torch.nn.Linear(embed_dim, embed_dim, bias=True)

        # self.fc = torch.nn.Linear(embed_dim, 1)

    def forward(self, query, key, value, attn_mask=None, need_weights=False):
        bsz, num_fields, embed_dim = query.size()

        q = self.linear_q(query)
        q = q.transpose(0, 1).contiguous()
        q = q.view(-1, bsz * self.num_heads, self.head_dim).transpose(0,
                                                                      1)
        q = q * self.scale
        k = self.linear_k(key)
        k = k.transpose(0, 1).contiguous()
        k = k.view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = value.transpose(0, 1).contiguous()
        v = v.view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))

        if attn_mask is not None:
            attn_output_weights += attn_mask

        attn_output_weights = torch.softmax(
            attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(attn_output_weights, self.dropout_p)

        attn_output = torch.bmm(attn_output_weights, v)
        assert list(attn_output.size()) == [bsz * self.num_heads, num_fields, self.head_dim]
        attn_output = attn_output.transpose(0, 1).contiguous().view(num_fields, bsz, embed_dim).transpose(0, 1)
        attn_output = self.output_layer(attn_output)
        if need_weights:
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, num_fields, num_fields)
            return attn_output, attn_output_weights.sum(dim=0) / bsz

        return attn_output, None


def get_activation_fn(activation: str):
    """ Returns the activation function corresponding to `activation` """
    if activation == "relu":
        return torch.relu
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation))