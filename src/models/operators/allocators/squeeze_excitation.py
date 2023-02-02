import re
from typing import Optional

from tensorflow.keras.layers import Layer
from tensorflow.keras.regularizers import Regularizer

from models.operators.layers import *
from .base import BaseOpAllocator


class SqueezeExcitationOpAllocator(BaseOpAllocator):
    def compile_op_regex(self) -> re.Pattern:
        return re.compile(rf'(?P<ratio>\d+)r SE')

    def generate_normal_layer(self, match: re.Match, filters: int, weight_reg: Optional[Regularizer], name_prefix: str = '',
                              name_suffix: str = '') -> Layer:
        ratio = int(match.group('ratio'))
        layer_name = f'{name_prefix}squeeze_excitation{name_suffix}'
        return SqueezeExcitation(self.op_dims, filters, ratio, use_bias=True, weight_reg=weight_reg, name=layer_name)

    def generate_depth_adaptation_layer(self, match: re.Match, filters: int, weight_reg: Optional[Regularizer], name_prefix: str = 'D/',
                                        name_suffix: str = '') -> Layer:
        ones = tuple([1] * self.op_dims)
        pointwise_conv = Convolution(filters, ones, ones, weight_reg=weight_reg)
        se_layer = self.generate_normal_layer(match, filters, weight_reg)

        layer_name = f'{name_prefix}pw_squeeze_excitation{name_suffix}'
        return ResizableSqueezeExcitation(se_layer, pointwise_conv, layer_name)

    def generate_spatial_adaptation_layer(self, match: re.Match, filters: int, weight_reg: Optional[Regularizer], strides: 'tuple[int, ...]',
                                          name_prefix: str = 'S/', name_suffix: str = '') -> Layer:
        pool_size = tuple([2] * self.op_dims)
        max_pool = Pooling('max', pool_size, pool_size)
        se_layer = self.generate_normal_layer(match, filters, weight_reg)

        layer_name = f'{name_prefix}pool_squeeze_excitation{name_suffix}'
        return ResizableSqueezeExcitation(se_layer, max_pool, layer_name)

    def generate_reduction_layer(self, match: re.Match, filters: int, weight_reg: Optional[Regularizer], strides: 'tuple[int, ...]',
                                 name_prefix: str = 'R/', name_suffix: str = '') -> Layer:
        pool_size = tuple([2] * self.op_dims)
        max_pool = Pooling('max', pool_size, pool_size)
        max_pool_pw = PoolingConv(max_pool, filters, weight_reg)
        se_layer = self.generate_normal_layer(match, filters, weight_reg)

        layer_name = f'{name_prefix}pool_pw__squeeze_excitation{name_suffix}'
        return ResizableSqueezeExcitation(se_layer, max_pool_pw, layer_name)