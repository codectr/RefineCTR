# -*- coding: UTF-8 -*-
"""
@project:RefineCTR
"""

import torch.nn as nn

from FRCTR.common import FeaturesLinear


class LogisticRegression(nn.Module):
    def __init__(self, field_dims):
        super(LogisticRegression, self).__init__()
        self.linear = FeaturesLinear(field_dims)

    def forward(self, x):
        pred_y = self.linear(x)
        return pred_y
