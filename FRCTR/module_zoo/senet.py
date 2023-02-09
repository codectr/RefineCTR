# -*- coding: UTF-8 -*-
"""
@project:RefineCTR
"""
import torch

from torch import nn


class SenetLayer(nn.Module):
    def __init__(self, field_length, ratio=1):
        super(SenetLayer, self).__init__()
        self.temp_dim = max(1, field_length // ratio)
        self.excitation = nn.Sequential(
            nn.Linear(field_length, self.temp_dim),
            nn.ReLU(),
            nn.Linear(self.temp_dim, field_length),
            nn.ReLU()
        )

    def forward(self, x_emb):
        """
        (1) Squeeze: max or mean
        (2) Excitation
        (3) Re-weight
        """
        Z_mean = torch.max(x_emb, dim=2, keepdim=True)[0].transpose(1, 2)
        # Z_mean = torch.mean(x_emb, dim=2, keepdim=True).transpose(1, 2)
        A_weight = self.excitation(Z_mean).transpose(1, 2)
        V_embed = torch.mul(A_weight, x_emb)
        return V_embed, A_weight
