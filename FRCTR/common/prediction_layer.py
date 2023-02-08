# -*- coding: UTF-8 -*-
"""
@project: RefineCTR
"""
import torch
from torch import nn



class BasicLR(nn.Module):
    def __init__(self, input_dim, sigmoid=False):
        super(BasicLR, self).__init__()
        self.sigmoid = sigmoid
        self.lr = nn.Linear(input_dim, 1, bias=True)

    def forward(self, x):
        return x


class BasicDNN(nn.Module):
    def __init__(self, input_dim, embed_dims, dropout=0.5, output_layer=True, sigmoid=False):
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

        self.sigmoid = sigmoid

    def _init_weight_(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        return self.mlp(x)