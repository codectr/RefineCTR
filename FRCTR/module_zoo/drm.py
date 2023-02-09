# -*- coding: UTF-8 -*-
"""
@project:RefineCTR
"""
import torch
from torch import nn
import torch.nn.functional as F


class DRMLayer(nn.Module):
    def __init__(self, field_length, scale=None):
        super(DRMLayer, self).__init__()
        self.trans_Q = nn.Linear(field_length, field_length)
        self.trans_K = nn.Linear(field_length, field_length)
        self.trans_V = nn.Linear(field_length, field_length)
        self.scale = scale

    def _field_attention(self, x_trans):
        Q = self.trans_Q(x_trans)
        K = self.trans_K(x_trans)
        V = self.trans_V(x_trans)

        attention = torch.matmul(Q, K.permute(0, 2, 1))
        if self.scale:
            attention = attention * self.scale
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context, attention

    def forward(self, x_emb):
        X_trans = x_emb.permute(0, 2, 1)  # B,E,F
        X_trans, att_score = self._field_attention(X_trans)
        X_trans = X_trans.permute(0, 2, 1) + x_emb  # B,F,E
        return X_trans.contiguous(), att_score
