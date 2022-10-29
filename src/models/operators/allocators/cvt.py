import re
from typing import Optional

from tensorflow.keras.layers import Layer
from tensorflow.keras.regularizers import Regularizer

from .base import BaseOpAllocator
from models.operators.layers import *


class CVTOpAllocator(BaseOpAllocator):
    def compile_op_regex(self) -> re.Pattern:
        return re.compile(r'(?P<kernel>\d+)k-(?P<heads>\d+)h-(?P<cvt_blocks>\d+)b cvt')

    def generate_reduction_layer(self, match: re.Match, filters: int, weight_reg: Optional[Regularizer], strides: 'tuple[int, ...]',
                                 name_prefix: str = 'R/', name_suffix: str = '') -> Layer:
        kernel = int(match.group('kernel'))
        heads = int(match.group('heads'))
        cvt_blocks = int(match.group('cvt_blocks'))
        layer_name = f'{name_prefix}{kernel}k-{heads}h-{cvt_blocks}b_cvt{name_suffix}'

        return CVTStage(emb_dim=filters, emb_kernel=kernel, emb_stride=strides[0], mlp_mult=2,
                        heads=heads, dim_head=filters, ct_blocks=cvt_blocks, weight_reg=weight_reg, name=layer_name)


class SimplifiedCVTOpAllocator(BaseOpAllocator):
    def compile_op_regex(self) -> re.Pattern:
        return re.compile(r'(?P<kernel>\d+)k-(?P<heads>\d+)h scvt')

    def generate_reduction_layer(self, match: re.Match, filters: int, weight_reg: Optional[Regularizer], strides: 'tuple[int, ...]',
                                 name_prefix: str = 'R/', name_suffix: str = '') -> Layer:
        kernel = int(match.group('kernel'))
        heads = int(match.group('heads'))
        layer_name = f'{name_prefix}{kernel}k-{heads}h_scvt{name_suffix}'

        return SimplifiedCVT(emb_dim=filters, emb_kernel=kernel, emb_stride=strides[0],
                             heads=heads, dim_head=filters, weight_reg=weight_reg, name=layer_name)