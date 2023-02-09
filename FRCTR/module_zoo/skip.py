# -*- coding: UTF-8 -*-
"""
@project:RefineCTR
"""
import torch.nn as nn

class Skip(nn.Module):
    def forward(self, x_emb):
        return x_emb, None