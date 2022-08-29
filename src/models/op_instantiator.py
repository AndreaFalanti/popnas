import re

from tensorflow.keras.layers import Layer, GlobalAveragePooling2D, GlobalAveragePooling1D, Add, Average
from tensorflow.keras.regularizers import Regularizer

from utils.func_utils import to_int_tuple
from .operators import *


class OpInstantiator:
    '''
    Class that takes care of building and returning valid Keras layers for the operators and input shape considered.
    Based on input shape, 1D or 2D operators are used.
    '''

    def __init__(self, input_dims: int, block_op_join: str, reduction_stride_factor: int = 2, weight_reg: Regularizer = None):
        self.weight_reg = weight_reg

        self.op_dims = input_dims - 1
        self.reduction_stride = tuple([reduction_stride_factor] * self.op_dims)
        self.normal_stride = tuple([1] * self.op_dims)

        gap_selector = {1: GlobalAveragePooling1D, 2: GlobalAveragePooling2D}
        self.gap = gap_selector[self.op_dims]

        self.block_join_op_selector = {'add': Add, 'avg': Average}
        self.block_op_join = block_op_join

        self.op_regexes = self.__compile_op_regexes()

        # enable Convolutional Vision Transformer only for images
        if input_dims == 3:
            self.op_regexes['cvt'] = re.compile(r'(\d+)k-(\d+)h-(\d+)b cvt')
            self.op_regexes['scvt'] = re.compile(r'(\d+)k-(\d+)h scvt')

    def __compile_op_regexes(self):
        '''
        Build a dictionary with compiled regexes for each parametrized supported operation.
        Adapt regexes based on input dimensionality.

        Returns:
            (dict): Regex dictionary
        '''
        # add groups to detect kernel size, based on op dimensionality.
        # e.g. Conv2D -> 3x3 conv, Conv1D -> 3 conv
        op_kernel_groups = 'x'.join([r'(\d+)'] * self.op_dims)

        return {'conv': re.compile(rf'{op_kernel_groups} conv'),
                'dconv': re.compile(rf'{op_kernel_groups} dconv'),
                'tconv': re.compile(rf'{op_kernel_groups} tconv'),
                'stack_conv': re.compile(rf'{op_kernel_groups}-{op_kernel_groups} conv'),
                'pool': re.compile(rf'{op_kernel_groups} (max|avg)pool')}

    def generate_block_join_operator(self, name_suffix: str):
        return self.block_join_op_selector[self.block_op_join](name=f'{self.block_op_join}{name_suffix}')

    def generate_pointwise_conv(self, filters: int, strided: bool, name: str):
        '''
        Provide builder for generating a pointwise convolution easily, for tensor shape regularization purposes.
        '''
        strides = self.reduction_stride if strided else self.normal_stride
        return Convolution(filters, tuple([1] * self.op_dims), strides=strides, name=name)

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

        # check non parametrized operations first since they don't require a regex and are faster
        if op_name == 'identity':
            # 'identity' action case, if using stride-2, then it's actually handled as a pointwise convolution
            if strided or adapt_depth:
                # TODO: IdentityReshaper leads to a strange non-deterministic bug and for now it has been disabled, reverting to pointwise convolution
                # layer_name = f'identity_reshaper{block_info_suffix}'
                # x = IdentityReshaper(filters, input_filters, strides, name=layer_name)
                layer_name = f'pointwise_id{layer_name_suffix}'
                kernel = tuple([1] * self.op_dims)
                return Convolution(filters, kernel, strides, weight_reg=self.weight_reg, name=layer_name)
            else:
                # else just submits a linear layer if shapes match
                layer_name = f'identity{layer_name_suffix}'
                return Identity(name=layer_name)

        # check for separable conv
        match = self.op_regexes['dconv'].match(op_name)  # type: re.Match
        if match:
            layer_name = f'{"x".join(match.groups())}_dconv{layer_name_suffix}'
            return SeparableConvolution(filters, kernel=to_int_tuple(match.groups()), strides=strides,
                                        name=layer_name, weight_reg=self.weight_reg)

        # check for transpose conv
        match = self.op_regexes['tconv'].match(op_name)  # type: re.Match
        if match:
            layer_name = f'{"x".join(match.groups())}_tconv{layer_name_suffix}'
            return TransposeConvolutionStack(filters, kernel=to_int_tuple(match.groups()), strides=strides,
                                             name=layer_name, weight_reg=self.weight_reg)

        # check for stacked conv operation
        match = self.op_regexes['stack_conv'].match(op_name)  # type: re.Match
        if match:
            g_count = len(match.groups())
            first_kernel_groups = match.groups()[:g_count // 2]
            second_kernel_groups = match.groups()[g_count // 2:]

            f = [filters, filters]
            k = [to_int_tuple(first_kernel_groups), to_int_tuple(second_kernel_groups)]
            s = [strides, self.normal_stride]

            layer_name = f'{"x".join(first_kernel_groups)}-{"x".join(second_kernel_groups)}_conv{layer_name_suffix}'
            return StackedConvolution(f, k, s, name=layer_name, weight_reg=self.weight_reg)

        # check for standard conv
        match = self.op_regexes['conv'].match(op_name)  # type: re.Match
        if match:
            layer_name = f'{"x".join(match.groups())}_conv{layer_name_suffix}'
            return Convolution(filters, kernel=to_int_tuple(match.groups()), strides=strides,
                               name=layer_name, weight_reg=self.weight_reg)

        # check for pooling
        match = self.op_regexes['pool'].match(op_name)  # type: re.Match
        if match:
            # last group is the pooling type
            pool_size = match.groups()[:-1]
            pool_type = match.groups()[-1]

            layer_name = f'{"x".join(pool_size)}_{pool_type}pool{layer_name_suffix}'
            return PoolingConv(filters, pool_type, to_int_tuple(pool_size), strides, name=layer_name, weight_reg=self.weight_reg) if adapt_depth \
                else Pooling(pool_type, to_int_tuple(pool_size), strides, name=layer_name)

        if self.op_dims == 2:
            match = self.op_regexes['cvt'].match(op_name)  # type: re.Match
            if match:
                layer_name = f'{match.group(1)}k-{match.group(2)}h-{match.group(3)}b_cvt{layer_name_suffix}'
                return CVTStage(emb_dim=filters, emb_kernel=int(match.group(1)), emb_stride=strides[0], mlp_mult=2,
                                heads=int(match.group(2)), dim_head=filters, ct_blocks=int(match.group(3)),
                                weight_reg=self.weight_reg, name=layer_name)

            match = self.op_regexes['scvt'].match(op_name)  # type: re.Match
            if match:
                layer_name = f'{match.group(1)}k-{match.group(2)}h_scvt{layer_name_suffix}'
                return SimplifiedCVT(emb_dim=filters, emb_kernel=int(match.group(1)), emb_stride=strides[0],
                                     heads=int(match.group(2)), dim_head=filters, weight_reg=self.weight_reg, name=layer_name)

        raise ValueError(f'Incorrect operator format or operator is not covered by POPNAS: {op_name}')
