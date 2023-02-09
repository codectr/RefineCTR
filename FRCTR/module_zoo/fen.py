# -*- coding: UTF-8 -*-
"""
@project:RefineCTR
"""
import torch
from torch import nn

class FENLayer(nn.Module):
    def __init__(self, field_length, embed_dim, mlp_layers=[256, 256, 256], h=1):
        super(FENLayer, self).__init__()
        self.h = h
        self.num_fields = field_length
        mlp_layers.append(self.num_fields)
        self.mlp_input_dim = self.num_fields * embed_dim
        self.mlp = MultiLayerPerceptron(self.mlp_input_dim, mlp_layers, dropout=0.5, output_layer=False)
        # self.lin_weight = nn.Linear(256,embed_dim,bias=False)

    def forward(self, x_emb):
        x_con = x_emb.view(-1, self.mlp_input_dim)  # B,F*E
        x_con = self.mlp(x_con)  # B,1
        x_weight = torch.softmax(x_con, dim=1) * self.h  # B,F
        x_emb_weight = x_emb * x_weight.unsqueeze(2)
        return x_emb_weight, x_weight

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