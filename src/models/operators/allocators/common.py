import re
from typing import Optional

from tensorflow.keras.layers import Layer
from tensorflow.keras.regularizers import Regularizer

from models.operators.layers import *
from models.operators.params_utils import compute_conv_params, compute_dconv_params
from .base import BaseOpAllocator, generate_kernel_group, opt_dilation_rate, regex_group_to_int_tuple, is_dilating_while_striding


class IdentityOpAllocator(BaseOpAllocator):
    def compile_op_regex(self) -> re.Pattern:
        return re.compile(r'identity')

    def compute_params(self, match: re.Match, input_filters: int, output_filters: int) -> int:
        # if identity needs to change shape, then it becomes a pointwise convolution
        return 0 if input_filters == output_filters else compute_conv_params((1, 1), input_filters, output_filters)

    def generate_normal_layer(self, match: re.Match, filters: int, weight_reg: Optional[Regularizer], name_prefix: str = '',
                              name_suffix: str = '') -> Layer:
        layer_name = f'{name_prefix}identity{name_suffix}'
        return Identity(name=layer_name)

    def generate_reduction_layer(self, match: re.Match, filters: int, weight_reg: Optional[Regularizer], strides: 'tuple[int, ...]',
                                 name_prefix: str = 'R/', name_suffix: str = '') -> Layer:
        layer_name = f'{name_prefix}pointwise_id{name_suffix}'
        pointwise_kernel = tuple([1] * self.op_dims)
        return Convolution(filters, pointwise_kernel, strides, weight_reg=weight_reg, name=layer_name)


class ConvolutionOpAllocator(BaseOpAllocator):
    def compile_op_regex(self) -> re.Pattern:
        return re.compile(rf'{generate_kernel_group(self.op_dims)}{opt_dilation_rate} conv')

    def compute_params(self, match: re.Match, input_filters: int, output_filters: int) -> int:
        return compute_conv_params(match.group('kernel'), input_filters, output_filters)

    def generate_reduction_layer(self, match: re.Match, filters: int, weight_reg: Optional[Regularizer], strides: 'tuple[int, ...]',
                                 name_prefix: str = 'R/', name_suffix: str = '') -> Layer:
        kernel_group = match.group('kernel')
        dilation_rate = 1 if match.group('dilation_rate') is None else int(match.group('dilation_rate'))

        dilation_subfix = '' if dilation_rate == 1 else f'_{dilation_rate}dr'
        layer_name = f'{name_prefix}{kernel_group}{dilation_subfix}_conv{name_suffix}'

        kernel = regex_group_to_int_tuple(kernel_group)
        layer = Convolution(filters, kernel=kernel, strides=strides, dilation_rate=dilation_rate, name=layer_name, weight_reg=weight_reg)

        return DilatedConvBatchActivationPooling(layer) if is_dilating_while_striding(dilation_rate, strides) else layer


class SeparableConvolutionOpAllocator(BaseOpAllocator):
    def compile_op_regex(self) -> re.Pattern:
        return re.compile(rf'{generate_kernel_group(self.op_dims)}{opt_dilation_rate} dconv')

    def compute_params(self, match: re.Match, input_filters: int, output_filters: int) -> int:
        return compute_dconv_params(match.group('kernel'), input_filters, output_filters)

    def generate_reduction_layer(self, match: re.Match, filters: int, weight_reg: Optional[Regularizer], strides: 'tuple[int, ...]',
                                 name_prefix: str = 'R/', name_suffix: str = '') -> Layer:
        kernel_group = match.group('kernel')
        dilation_rate = 1 if match.group('dilation_rate') is None else int(match.group('dilation_rate'))

        dilation_subfix = '' if dilation_rate == 1 else f'_{dilation_rate}dr'
        layer_name = f'{name_prefix}{kernel_group}{dilation_subfix}_dconv{name_suffix}'

        kernel = regex_group_to_int_tuple(kernel_group)
        layer = SeparableConvolution(filters, kernel=kernel, strides=strides, dilation_rate=dilation_rate, name=layer_name, weight_reg=weight_reg)

        return DilatedConvBatchActivationPooling(layer) if is_dilating_while_striding(dilation_rate, strides) else layer


class StackedConvolutionOpAllocator(BaseOpAllocator):
    def compile_op_regex(self) -> re.Pattern:
        return re.compile(rf'{generate_kernel_group(self.op_dims, "_1")}-{generate_kernel_group(self.op_dims, "_2")} conv')

    def compute_params(self, match: re.Match, input_filters: int, output_filters: int) -> int:
        return compute_conv_params(match.group('kernel_1'), input_filters, output_filters) + \
            compute_conv_params(match.group('kernel_2'), output_filters, output_filters)

    def generate_reduction_layer(self, match: re.Match, filters: int, weight_reg: Optional[Regularizer], strides: 'tuple[int, ...]',
                                 name_prefix: str = 'R/', name_suffix: str = '') -> StackedConvolution:
        kernel_group_1 = match.group('kernel_1')
        kernel_group_2 = match.group('kernel_2')

        layer_name = f'{name_prefix}{kernel_group_1}-{kernel_group_2}_conv{name_suffix}'

        kernel_1 = regex_group_to_int_tuple(kernel_group_1)
        kernel_2 = regex_group_to_int_tuple(kernel_group_2)

        f = [filters] * 2
        k = [kernel_1, kernel_2]
        s = [strides, tuple([1] * len(strides))]

        return StackedConvolution(f, k, s, name=layer_name, weight_reg=weight_reg)


class TransposeConvolutionOpAllocator(BaseOpAllocator):
    def compile_op_regex(self) -> re.Pattern:
        return re.compile(rf'{generate_kernel_group(self.op_dims)} tconv')

    def compute_params(self, match: re.Match, input_filters: int, output_filters: int) -> int:
        return compute_conv_params(match.group('kernel'), input_filters, output_filters) + \
            compute_conv_params(match.group('kernel'), output_filters, output_filters)

    def generate_reduction_layer(self, match: re.Match, filters: int, weight_reg: Optional[Regularizer], strides: 'tuple[int, ...]',
                                 name_prefix: str = 'R/', name_suffix: str = '') -> TransposeConvolutionStack:
        kernel_group = match.group('kernel')
        layer_name = f'{name_prefix}{kernel_group}_tconv{name_suffix}'

        kernel = regex_group_to_int_tuple(kernel_group)
        return TransposeConvolutionStack(filters, kernel=kernel, strides=strides, name=layer_name, weight_reg=weight_reg)


class PoolOpAllocator(BaseOpAllocator):
    def compile_op_regex(self) -> re.Pattern:
        return re.compile(rf'{generate_kernel_group(self.op_dims)} (?P<type>max|avg)pool')

    def compute_params(self, match: re.Match, input_filters: int, output_filters: int) -> int:
        # if pooling needs to adapt the filters, then it is followed by a pointwise convolution
        return 0 if input_filters == output_filters else compute_conv_params((1, 1), input_filters, output_filters)

    def generate_normal_layer(self, match: re.Match, filters: int, weight_reg: Optional[Regularizer], name_prefix: str = '',
                              name_suffix: str = '') -> Layer:
        return self.generate_spatial_adaptation_layer(match, filters, weight_reg, tuple([1] * self.op_dims), name_prefix, name_suffix)

    def generate_spatial_adaptation_layer(self, match: re.Match, filters: int, weight_reg: Optional[Regularizer], strides: 'tuple[int, ...]',
                                          name_prefix: str = 'S/', name_suffix: str = '') -> Pooling:
        size_group = match.group('kernel')  # named kernel in regex but proper name is size
        pool_type = match.group('type')
        layer_name = f'{name_prefix}{size_group}_{pool_type}pool{name_suffix}'

        pool_size = regex_group_to_int_tuple(size_group)
        return Pooling(pool_type, pool_size, strides, name=layer_name)

    def generate_reduction_layer(self, match: re.Match, filters: int, weight_reg: Optional[Regularizer], strides: 'tuple[int, ...]',
                                 name_prefix: str = 'R/', name_suffix: str = '') -> PoolingConv:
        pool_layer = self.generate_spatial_adaptation_layer(match, filters, weight_reg, strides, name_prefix, name_suffix)
        return PoolingConv(pool_layer, filters, weight_reg=weight_reg)

