import re

from tensorflow.keras.layers import Layer

from models.operators.layers import *
from models.operators.params_utils import compute_conv_params
from .base import BaseOpAllocator, generate_kernel_group, regex_group_to_int_tuple


class ZoomedConvolutionOpAllocator(BaseOpAllocator):
    def compile_op_regex(self) -> re.Pattern:
        return re.compile(rf'{generate_kernel_group(self.op_dims)}:(?P<zoom_factor>\d+)z zconv')

    def compute_params(self, match: re.Match, input_filters: int, output_filters: int) -> int:
        return compute_conv_params(match.group('kernel'), input_filters, output_filters)

    def generate_reduction_layer(self, match: re.Match, filters: int, strides: 'tuple[int, ...]',
                                 name_prefix: str = 'R/', name_suffix: str = '') -> Layer:
        kernel_group = match.group('kernel')
        kernel = regex_group_to_int_tuple(kernel_group)
        zoom_factor = int(match.group('zoom_factor'))

        layer_name = f'{name_prefix}{kernel_group}_{zoom_factor}z_zconv{name_suffix}'

        return ZoomedConvolution(filters, kernel=kernel, strides=strides, zoom_factor=zoom_factor,
                                 weight_reg=self.weight_reg, activation_f=self.activation_f, name=layer_name)
