# -*- coding: UTF-8 -*-
"""
@project: RefineCTR
"""
import sys
sys.path.append("../")

from .fm import FMFrn
from .deepfm import DeepFMFrn, DeepFMFrnP
from .dcn import CNFrn, DCNFrn, DCNFrnP
from .dcnv2 import CN2Frn, DCNV2Frn, DCNV2FrnP
from .afnp import AFNFrn, AFNPlusFrn, AFNPlusFrnP
from .xdeepfm import CINFrn, xDeepFMFrn, xDeepFMFrnP

from .fibinet import FiBiNetFrn
from .fwfm import FwFMFrn
from .fnn import FNNFrn
from .nfm import NFMFrn
from .fint import FINTFrn