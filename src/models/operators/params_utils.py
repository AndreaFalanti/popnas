from typing import Union

from utils.func_utils import to_int_tuple, prod


def compute_batch_norm_params(filters: int):
    return 4 * filters


def compute_layer_norm_params(filters: int):
    # beta and gamma for each channel (since layer norm is applied on axis -1)
    return 2 * filters


def compute_conv_params(kernel: Union[str, 'tuple[Union[str, int], ...]'], filters_in: int, filters_out: int, bias: bool = True, bn: bool = True):
    '''
    Formula for computing the parameters of many convolutional operators.

    Args:
        kernel: kernel size
        filters_in: input filters
        filters_out: output filters
        bias: use bias or not
        bn: followed by BatchNormalization or not
    '''
    # split kernel in multiple digits, if in str format
    if isinstance(kernel, str):
        kernel = kernel.split('x')

    kernel = to_int_tuple(kernel)  # cast to int in case are elements are str
    b = 1 if bias else 0
    bn_params = compute_batch_norm_params(filters_out) if bn else 0

    return (prod(kernel) * filters_in + b) * filters_out + bn_params


def compute_dconv_params(kernel: Union[str, 'tuple[Union[str, int], ...]'], filters_in: int, filters_out: int, bias: bool = True, bn: bool = True):
    ''' Depthwise separable convolution + batch norm. '''
    # bias term is used only in pointwise for unknown reasons, also it has no batch normalization,
    # so it is computed separately without "compute_conv_params" function.

    # split kernel in multiple digits, if in str format
    if isinstance(kernel, str):
        kernel = kernel.split('x')

    kernel = to_int_tuple(kernel)  # cast to int in case are elements are str
    return (prod(kernel) * filters_in) + \
           compute_conv_params((1, 1), filters_in, filters_out, bias=bias, bn=bn)


def compute_cvt_params(kernel: 'tuple[Union[str, int], ...]', heads: int, blocks: int, filters_in: int, filters_out: int, use_mlp: bool):
    ''' Helper function for computing the params of a Convolutional Transformer block (CVT). '''
    # +1 is bias term
    kernel = to_int_tuple(kernel)  # cast to int in case are elements are str
    dim_head = filters_out  # forced by actual implementation (see op_instantiator)

    embed_conv_params = compute_conv_params(kernel, filters_in, filters_out, bn=False)
    layer_norm_params = compute_layer_norm_params(filters_out)

    # NOTE: dconv here actually use bn, but is in the middle instead of the end, so they use the initial filters as num of channels!
    # done separately to avoid error in computation
    q_conv_params = compute_dconv_params((3, 3), filters_out, dim_head * heads, bias=False, bn=False)
    kv_conv_params = compute_dconv_params((3, 3), filters_out, 2 * dim_head * heads, bias=False, bn=False)
    bn_params = compute_batch_norm_params(filters_out) * 2
    conv_out = compute_conv_params((1, 1), dim_head * heads, filters_out, bn=False)

    mlp_mult = 2
    mlp_params = (2 * layer_norm_params +
                  compute_conv_params((1, 1), filters_out, filters_out * mlp_mult, bn=False) +
                  compute_conv_params((1, 1), filters_out * mlp_mult, filters_out, bn=False)) if use_mlp else 0

    ct_block_params = q_conv_params + kv_conv_params + bn_params + conv_out + mlp_params

    return embed_conv_params + layer_norm_params + ct_block_params * blocks
