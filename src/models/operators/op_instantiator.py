from typing import Callable

from tensorflow.keras import activations
from tensorflow.keras.layers import Layer, Add, Average
from tensorflow.keras.regularizers import Regularizer

from models.operators.allocators import *
from models.operators.layers import Convolution, TransposeConvolution, op_dim_selector


class OpInstantiator:
    '''
    Class that takes care of building and returning valid Keras layers for the operators and input shape considered.
    Based on input shape, 1D or 2D operators are used accordingly.
    '''

    def __init__(self, input_dims: int, block_op_join: str, reduction_stride_factor: int = 2,
                 weight_reg: Regularizer = None, activation_f: Callable = activations.swish):
        self.weight_reg = weight_reg
        self.activation_f = activation_f

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
            'identity': IdentityOpAllocator(self.op_dims, self.weight_reg, self.activation_f),
            'conv': ConvolutionOpAllocator(self.op_dims, self.weight_reg, self.activation_f),
            'dconv': SeparableConvolutionOpAllocator(self.op_dims, self.weight_reg, self.activation_f),
            'tconv': TransposeConvolutionOpAllocator(self.op_dims, self.weight_reg, self.activation_f),
            'stack_conv': StackedConvolutionOpAllocator(self.op_dims, self.weight_reg, self.activation_f),
            'pool': PoolOpAllocator(self.op_dims, self.weight_reg, self.activation_f),
            'se': SqueezeExcitationOpAllocator(self.op_dims, self.weight_reg, self.activation_f)
        }

    def _define_2d_specific_allocators(self) -> 'dict[str, BaseOpAllocator]':
        return {
            'cvt': CVTOpAllocator(self.op_dims, self.weight_reg, self.activation_f),
            'scvt': SimplifiedCVTOpAllocator(self.op_dims, self.weight_reg, self.activation_f)
        }

    def _define_1d_specific_allocators(self) -> 'dict[str, BaseOpAllocator]':
        return {
            'lstm': LSTMOpAllocator(self.op_dims, self.weight_reg, self.activation_f),
            'gru': GRUOpAllocator(self.op_dims, self.weight_reg, self.activation_f)
        }

    def generate_block_join_operator(self, name_suffix: str):
        return self.block_join_op_selector[self.block_op_join](name=f'{self.block_op_join}{name_suffix}')

    def generate_pointwise_conv(self, filters: int, strided: bool, name: str):
        '''
        Provide builder for easily generating a pointwise convolution, for tensor shape regularization purposes.
        '''
        strides = self.reduction_stride if strided else self.normal_stride
        return Convolution(filters, tuple([1] * self.op_dims), strides=strides, activation_f=self.activation_f, name=name)

    def generate_transpose_conv(self, filters: int, upsample_factor: int, name: str):
        '''
        Provide builder for easily generating a transpose convolution, for tensor shape regularization purposes in specific networks types,
        e.g, segmentation networks, which could require a way of upsampling the spatial resolution of a tensor.
        '''
        kernel = tuple([upsample_factor] * self.op_dims)
        stride = tuple([upsample_factor] * self.op_dims)
        return TransposeConvolution(filters, kernel, stride, weight_reg=self.weight_reg, activation_f=self.activation_f, name=name)

    def generate_linear_upsample(self, upsample_factor: int, name: str):
        upsampler = op_dim_selector['upsample'][self.op_dims]
        interpolation_type = 'bilinear' if self.op_dims == 2 else 'linear'
        # TODO: currently Upsample1D is the one implemented in Keras, which does not support linear upsample.
        #  Refer to this issue for extending it: https://github.com/tensorflow/tensorflow/issues/46609
        return upsampler(upsample_factor, interpolation=interpolation_type, name=name)

    def build_op_layer(self, op_name: str, filters: int, layer_name_suffix: str, adapt_spatial: bool = False, adapt_depth: bool = False) -> Layer:
        '''
        Generate a custom Keras layer for the provided operator and parameter. Certain operations are handled in a different way
        when used in reduction cells, compared to the normal cells, to handle the tensor shape changes and allow addition at the end of a block.

        # Args:
            op_name: operator to use
            filters: number of filters
            input_filters: number of filters of the input of this new layer
            layer_name_suffix: suffix appended to layer name, basically unique metadata about block and cell indexes
            adapt_spatial: if op must adapt the tensor spatial dimensionality (commonly by using a stride different from 1)
            adapt_depth: if op must adapt the tensor channels

        # Returns:
            (tf.keras.layers.Layer): The custom layer corresponding to the operator
        '''
        strides = self.reduction_stride if adapt_spatial else self.normal_stride

        # iterate on allocators, stop at first match and allocates the Keras layer based on conditions
        for allocator in self.op_allocators.values():
            match = allocator.is_match(op_name)
            if match:
                if adapt_depth and adapt_spatial:
                    return allocator.generate_reduction_layer(match, filters, strides, name_suffix=layer_name_suffix)
                elif adapt_spatial:
                    return allocator.generate_spatial_adaptation_layer(match, filters, strides, name_suffix=layer_name_suffix)
                elif adapt_depth:
                    return allocator.generate_depth_adaptation_layer(match, filters, name_suffix=layer_name_suffix)
                else:
                    return allocator.generate_normal_layer(match, filters, name_suffix=layer_name_suffix)

        # raised only if no allocator matches the operator string
        raise ValueError(f'Incorrect operator format or operator is not covered by POPNAS: {op_name}')

    def get_op_params(self, op_name: str, input_shape: 'list[float]', output_shape: 'list[float]'):
        input_filters = int(input_shape[-1])
        output_filters = int(output_shape[-1])

        # these operators can't have parameters in any case and are not covered in allocators
        no_params_operators = ['add', 'concat', 'input', 'gap']
        if op_name in no_params_operators:
            return 0

        for allocator in self.op_allocators.values():
            match = allocator.is_match(op_name)
            if match:
                return allocator.compute_params(match, input_filters, output_filters)

        # raised only if no allocator matches the operator string
        raise ValueError(f'Incorrect operator format or operator is not covered by POPNAS: {op_name}')
