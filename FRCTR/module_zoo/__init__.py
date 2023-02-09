# -*- coding: UTF-8 -*-
"""
@project: RefineCTR
"""

import torch

# import sys
# sys.path.append("../")

from .skip import Skip
from .contextnet import TCELayer, PFFNLayer
from .dfen import DualFENLayer
from .fen import FENLayer
from .drm import DRMLayer
from .frnet import FRNetLayer
from .fwn import FWNLayer
from .gatenet import GateLayer
from .selfatt import InterCTRLayer
from .senet import SenetLayer
from .gfrllayer import GFRLLayer

ALLFrn_OPS = {
    "skip": lambda field_length, embed_dim: Skip(),  # Skip-connection
    "senet": lambda field_length, embed_dim: SenetLayer(field_length, ratio=2),
    "fen": lambda field_length, embed_dim: FENLayer(field_length, embed_dim, mlp_layers=[256, 256, 256]),
    "non": lambda field_length, embed_dim: FWNLayer(field_length, embed_dim),
    "drm": lambda field_length, embed_dim: DRMLayer(field_length),
    "dfen": lambda field_length, embed_dim: DualFENLayer(field_length, embed_dim, att_size=embed_dim,
                                                         num_heads=4, embed_dims=[256, 256, 256]),
    "gate_v": lambda field_length, embed_dim: GateLayer(field_length, embed_dim, gate_type="vec"),
    "gate_b": lambda field_length, embed_dim: GateLayer(field_length, embed_dim, gate_type="bit"),
    "pffn": lambda field_length, embed_dim: PFFNLayer(field_length, embed_dim, project_dim=32, num_blocks=3),
    "tce": lambda field_length, embed_dim: TCELayer(field_length, embed_dim, project_dim=2*embed_dim),
    "gfrl": lambda field_length, embed_dim: GFRLLayer(field_length, embed_dim, dnn_size=[256]),
    "frnet_v": lambda field_length, embed_dim: FRNetLayer(field_length, embed_dim, weight_type="vec",
                                                          num_layers=1, att_size=16, mlp_layer=128),
    "frnet_b": lambda field_length, embed_dim: FRNetLayer(field_length, embed_dim, weight_type="bit",
                                                          num_layers=1, att_size=16, mlp_layer=128),
    "selfatt": lambda field_length, embed_dim: InterCTRLayer(embed_dim, att_size=16,
                                                             num_heads=8, out_dim=embed_dim)
}

if __name__ == '__main__':
    inputs = torch.randn(10, 20, 16)
    names = ["skip", "drm","non","senet","fen", "dfen","selfatt", "frnet_b", "frnet_v", "gfrl", "tce", "pffn", "gate_b", "gate_v"]
    for index, name in enumerate(names):
        frn = ALLFrn_OPS[name](20, 16)
        out, weight = frn(inputs)
        print("index:{}, frn:{}, size:{}".format(index+1, name, out.size()))
