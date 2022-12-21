from typing import Callable

import tensorflow as tf
from tensorflow.keras.layers import Layer, Add, Average
from tensorflow.keras.regularizers import Regularizer

from models.operators.allocators import *
from models.operators.layers import Convolution, op_dim_selector


class OpInstantiator:
    '''
    Class that takes care of building and returning valid Keras layers for the operators and input shape considered.
    Based on input shape, 1D or 2D operators are used accordingly.
    '''

    def __init__(self, input_dims: int, block_op_join: str, reduction_stride_factor: int = 2, weight_reg: Regularizer = None):
        self.weight_reg = weight_reg

        self.op_dims = input_dims - 1
        self.reduction_stride = tuple([reduction_stride_factor] * self.op_dims)
        self.normal_stride = tuple([1] * self.op_dims)

        self.gap = op_dim_selector['gap'][self.op_dims]

        self.block_join_op_selector = {'add': Add, 'avg': Average}
        self.block_op_join = block_op_join

        self.op_allocators = self._define_common_allocators()

        # enable operators available only for images
        if self.op_dims == 2:
            self.op_allocators.update(self._define_2d_specific_allocators())
        elif self.op_dims == 1:
            self.op_allocators.update(self._define_1d_specific_allocators())

    def _define_common_allocators(self) -> 'dict[str, BaseOpAllocator]':
        return {
            'identity': IdentityOpAllocator(self.op_dims),
            'conv': ConvolutionOpAllocator(self.op_dims),
            'dconv': SeparableConvolutionOpAllocator(self.op_dims),
            'tconv': TransposeConvolutionOpAllocator(self.op_dims),
            'stack_conv': StackedConvolutionOpAllocator(self.op_dims),
            'pool': PoolOpAllocator(self.op_dims)
        }

    def _define_2d_specific_allocators(self) -> 'dict[str, BaseOpAllocator]':
        return {
            'cvt': CVTOpAllocator(self.op_dims),
            'scvt': SimplifiedCVTOpAllocator(self.op_dims)
        }

    def _define_1d_specific_allocators(self) -> 'dict[str, BaseOpAllocator]':
        return {
            'lstm': LSTMOpAllocator(self.op_dims),
            'gru': GRUOpAllocator(self.op_dims)
        }

    def generate_block_join_operator(self, name_suffix: str):
        return self.block_join_op_selector[self.block_op_join](name=f'{self.block_op_join}{name_suffix}')

    def generate_pointwise_conv(self, filters: int, strided: bool, name: str, activation_f: Callable = tf.nn.silu):
        '''
        Provide builder for generating a pointwise convolution easily, for tensor shape regularization purposes.
        '''
        strides = self.reduction_stride if strided else self.normal_stride
        return Convolution(filters, tuple([1] * self.op_dims), strides=strides, activation_f=activation_f, name=name)

    def build_op_layer(self, op_name: str, filters: int, input_filters: int, layer_name_suffix: str, strided: bool = False) -> Layer:
        '''
        Generate a custom Keras layer for the provided operator and parameter. Certain operations are handled in a different way
        when used in reduction cells, compared to the normal cells, to handle the tensor shape changes and allow addition at the end of a block.

        # Args:
            op_name: operator to use
            filters: number of filters
            input_filters: number of filters of the input of this new layer
            layer_name_suffix: suffix appended to layer name, basically unique metadata about block and cell indexes
            strided: if op must use a stride different from 1 (reduction cells)

        # Returns:
            (tf.keras.layers.Layer): The custom layer corresponding to the operator
        '''
        adapt_depth = filters != input_filters
        strides = self.reduction_stride if strided else self.normal_stride

        # iterate on allocators, stop at first match and allocates the Keras layer based on conditions
        for allocator in self.op_allocators.values():
            match = allocator.is_match(op_name)
            if match:
                if adapt_depth and strided:
                    return allocator.generate_reduction_layer(match, filters, self.weight_reg, strides, name_suffix=layer_name_suffix)
                elif strided:
                    return allocator.generate_spatial_adaptation_layer(match, filters, self.weight_reg, strides, name_suffix=layer_name_suffix)
                elif adapt_depth:
                    return allocator.generate_depth_adaptation_layer(match, filters, self.weight_reg, name_suffix=layer_name_suffix)
                else:
                    return allocator.generate_normal_layer(match, filters, self.weight_reg, name_suffix=layer_name_suffix)

        # raised only if no allocator matches the operator string
        raise ValueError(f'Incorrect operator format or operator is not covered by POPNAS: {op_name}')
