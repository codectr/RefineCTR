# -*- coding: UTF-8 -*-
"""
@project:RefineCTR
"""
import torch
import torch.nn.functional as F
from torch import nn


class TCELayer(nn.Module):
    def __init__(self, field_length, embed_dim, project_dim=32, agg_share=False):
        super(TCELayer, self).__init__()
        # sharing parameter in aggregation layer
        input_dim = field_length * embed_dim
        if agg_share:
            # individual parameter for each feature(field)
            self.aggregation_w = nn.Parameter(torch.randn(1, input_dim, project_dim))
        else:
            # share nothing
            self.aggregation_w = nn.Parameter(torch.randn(field_length, input_dim, project_dim))

        self.field_length = field_length
        self.project_w = nn.Parameter(torch.randn(field_length, project_dim, embed_dim))
        nn.init.xavier_uniform_(self.project_w.data)
        nn.init.xavier_uniform_(self.aggregation_w.data)

    def forward(self, x_emb):
        x_cat = x_emb.view(x_emb.size(0), -1).unsqueeze(0).expand(self.field_length, -1, -1)  # F,B,F*E
        x_agg = torch.relu(torch.matmul(x_cat, self.aggregation_w))  # F, B, P
        x_project = torch.matmul(x_agg, self.project_w).permute(1, 0, 2)  # FBP FPE = FBE => B,F,E
        x_emb = x_emb * torch.relu(x_project)
        return x_emb, x_project


class PFFNLayer(nn.Module):
    def __init__(self, field_length, embed_dim, project_dim=32,
                 agg_share=False, num_blocks=3):
        super(PFFNLayer, self).__init__()
        self.tce_layer = TCELayer(field_length, embed_dim, project_dim, agg_share=agg_share)
        # Do not share any parameter in Point-wise FFN:
        self.W1 = nn.Parameter(torch.randn(field_length, embed_dim, embed_dim))

        # Sharing the parameters
        # self.W1 = nn.Parameter(torch.randn(1, embed_dim, embed_dim))
        # self.W2 = nn.Parameter(torch.randn(1, embed_dim, embed_dim))
        self.LN = nn.LayerNorm(embed_dim)
        self.num_blocks = num_blocks
        nn.init.xavier_uniform_(self.W1.data)
        # nn.init.xavier_uniform_(self.W2.data)

    def forward(self, x_emb):
        x_emb = self.tce_layer(x_emb)[0]
        for _ in range(self.num_blocks):
            x_emb_ = torch.matmul(x_emb.permute(1, 0, 2), self.W1)  # F,B,E
            x_emb = self.LN(x_emb_.permute(1, 0, 2) + x_emb)
            # x_emb_ = torch.relu(torch.matmul(x_emb, self.W1))  # F,B,E
            # x_emb_ = torch.matmul(x_emb_, self.W2)
            # x_emb = self.LN(x_emb_ + x_emb)  # ,B,F,E
        return x_emb, None


class SFFN(nn.Module):
    def __init__(self, field_length, embed_dim, project_dim=32,
                 agg_share=False, num_blocks=3):
        super(SFFN, self).__init__()
        self.tce_layer = TCELayer(field_length, embed_dim, project_dim, agg_share=agg_share)
        self.W1 = nn.Parameter(torch.randn(1, embed_dim, embed_dim))
        self.LN = nn.LayerNorm(embed_dim)
        self.num_blocks = num_blocks
        nn.init.xavier_uniform_(self.W1.data)

    def forward(self, x_emb):
        x_emb = self.tce_layer(x_emb)[0]
        for _ in range(self.num_blocks):
            x_emb = self.LN(torch.matmul(x_emb, self.W1))
        return x_emb, None
