# -*- coding: UTF-8 -*-
"""
@project: RefineCTR
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BasicFRCTR(nn.Module):
    def __init__(self, field_dims, embed_dim, FRN=None):
        super(BasicFRCTR, self).__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        if not FRN:
            raise ValueError("Feature Refinement Module is None")
        self.frn = FRN
        self.num_fields = len(field_dims)

    def forward(self, x):
        raise NotImplemented


class FeaturesLinear(nn.Module):
    def __init__(self, field_dims, output_dim=1):
        super().__init__()
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)

    def forward(self, x):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return torch.sum(self.fc(x), dim=1) + self.bias


class FeaturesEmbedding(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)
