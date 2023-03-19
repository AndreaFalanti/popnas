import re

from tensorflow.keras.layers import Layer

from models.operators.layers import *
from models.operators.params_utils import compute_cvt_params
from .base import BaseOpAllocator


class CVTOpAllocator(BaseOpAllocator):
    def compile_op_regex(self) -> re.Pattern:
        return re.compile(r'(?P<kernel>\d+)k-(?P<heads>\d+)h-(?P<cvt_blocks>\d+)b cvt')

    def compute_params(self, match: re.Match, input_filters: int, output_filters: int) -> int:
        k, heads, blocks = int(match.group(1)), int(match.group(2)), int(match.group(3))
        return compute_cvt_params((k, k), heads, blocks, input_filters, output_filters, use_mlp=True)

    def generate_reduction_layer(self, match: re.Match, filters: int, strides: 'tuple[int, ...]',
                                 name_prefix: str = 'R/', name_suffix: str = '') -> Layer:
        kernel = int(match.group('kernel'))
        heads = int(match.group('heads'))
        cvt_blocks = int(match.group('cvt_blocks'))
        layer_name = f'{name_prefix}{kernel}k-{heads}h-{cvt_blocks}b_cvt{name_suffix}'

        return CVTStage(emb_dim=filters, emb_kernel=kernel, emb_stride=strides[0], mlp_mult=2,
                        heads=heads, dim_head=filters, ct_blocks=cvt_blocks, weight_reg=self.weight_reg, name=layer_name)


class SimplifiedCVTOpAllocator(BaseOpAllocator):
    def compile_op_regex(self) -> re.Pattern:
        return re.compile(r'(?P<kernel>\d+)k-(?P<heads>\d+)h scvt')

    def compute_params(self, match: re.Match, input_filters: int, output_filters: int) -> int:
        k, heads = int(match.group(1)), int(match.group(2))
        return compute_cvt_params((k, k), heads, 1, input_filters, output_filters, use_mlp=False)

    def generate_reduction_layer(self, match: re.Match, filters: int, strides: 'tuple[int, ...]',
                                 name_prefix: str = 'R/', name_suffix: str = '') -> Layer:
        kernel = int(match.group('kernel'))
        heads = int(match.group('heads'))
        layer_name = f'{name_prefix}{kernel}k-{heads}h_scvt{name_suffix}'

        return SimplifiedCVT(emb_dim=filters, emb_kernel=kernel, emb_stride=strides[0],
                             heads=heads, dim_head=filters, weight_reg=self.weight_reg, name=layer_name)
