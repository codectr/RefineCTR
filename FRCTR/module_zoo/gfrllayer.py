# -*- coding: UTF-8 -*-
"""
@project:RefineCTR
"""
import torch
from torch import nn

class GFRLLayer(nn.Module):
    def __init__(self, field_length, embed_dim, dnn_size=[256]):
        super(GFRLLayer, self).__init__()
        self.flu1 = FLU(field_length, embed_dim, dnn_size=dnn_size)
        self.flu2 = FLU(field_length, embed_dim, dnn_size=dnn_size)

    def forward(self, x_emb):
        x_out = self.flu1(x_emb)
        x_pro = torch.sigmoid(self.flu2(x_emb))
        x_out = x_emb * (torch.tensor(1.0) - x_pro) + x_out * x_pro
        return x_out, x_pro


class FLU(nn.Module):
    def __init__(self, field_length, embed_dim, dnn_size=[256]):
        super(FLU, self).__init__()
        self.input_dim = field_length * embed_dim
        self.local_w = nn.Parameter(torch.randn(field_length, embed_dim, embed_dim))
        self.local_b = nn.Parameter(torch.randn(field_length, 1, embed_dim))

        self.mlps = MultiLayerPerceptron(self.input_dim, embed_dims=dnn_size, output_layer=False)
        self.bit_info = nn.Linear(dnn_size[-1], embed_dim)
        self.acti = nn.ReLU()

        nn.init.xavier_uniform_(self.local_w.data)
        nn.init.xavier_uniform_(self.local_b.data)

    def forward(self, x_emb):
        x_local = torch.matmul(x_emb.permute(1, 0, 2), self.local_w) + self.local_b
        x_local = x_local.permute(1, 0, 2)  # B,F,E

        x_glo = self.mlps(x_emb.view(-1, self.input_dim))
        x_glo = self.acti(self.bit_info(x_glo)).unsqueeze(1)  # B, E
        x_out = x_local * x_glo
        return x_out

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