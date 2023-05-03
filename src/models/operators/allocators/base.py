import re
from abc import abstractmethod
from typing import Optional, Callable

from tensorflow.keras.layers import Layer
from tensorflow.keras.regularizers import Regularizer

from utils.func_utils import to_int_tuple

# common regex groups for parameters and utilities. They can be imported by Allocator concrete implementations.
opt_dilation_rate = r'(:(?P<dilation_rate>\d+)dr)?'


def generate_kernel_group(op_dims: int, group_suffix: str = '') -> str:
    kernel_group = 'x'.join([r'\d+'] * op_dims)
    return rf'(?P<kernel{group_suffix}>{kernel_group})'


def regex_group_to_int_tuple(regex_group: str, number_sep: str = 'x'):
    '''
    Parse a regex string composed of multiple numbers separated by given char, returning a tuple of integers.
    '''
    return to_int_tuple(regex_group.split(number_sep))


def is_dilating_while_striding(dilation_rate: int, strides: 'tuple[int, ...]'):
    '''
    Checks if both dilation and strides are > 1.
    Since TensorFlow does not support using both, this special case must be addressed with different solutions.
    '''
    return dilation_rate > 1 and any(map(lambda s: s > 1, strides))


class BaseOpAllocator:
    '''
    Base abstract class for implementing operator allocators.

    An operator allocator defines the logic for generating the layers associated to a POPNAS operator, so they are necessary for any operator
    supported in the configuration's operator set.

    POPNAS operators must be able to change the spatial and depth dimensions, to adapt to the network structure.
    The allocator API handles all reshaping cases: the functions necessary for the operator special cases + reduction must be specified
    in the concrete implementation.
    '''
    def __init__(self, op_dims: int, weight_reg: Optional[Regularizer], activation_f: Callable) -> None:
        '''
        Args:
            op_dims: int representing the dimensional domain of the operator (1 for time series, 2 for images)
            weight_reg: an optional Keras regularizer to apply on weights
            activation_f: the activation function to use when not fixed by layers definition
        '''
        self.op_dims = op_dims
        self.weight_reg = weight_reg
        self.activation_f = activation_f

        self.regex = self.compile_op_regex()

    def is_match(self, op_name: str):
        '''
        Checks if the operator name matches the allocator regex structure.

        Args:
            op_name: operator name provided in the POPNAS configuration operator set

        Returns:
            the regex Match object, None if not matching
        '''
        return self.regex.match(op_name)

    @abstractmethod
    def compile_op_regex(self) -> re.Pattern:
        ''' Compile the regex that matches the operator. If the operator can be parametrized, regex groups must be defined accordingly. '''
        raise NotImplementedError()

    @abstractmethod
    def compute_params(self, match: re.Match, input_filters: int, output_filters: int) -> int:
        raise NotImplementedError()

    def generate_normal_layer(self, match: re.Match, filters: int, name_prefix: str = '', name_suffix: str = '') -> Layer:
        '''
        Generate a normal layer for the operator (no spatial or depth dimensionality changes).
        If not overridden, then the function generates the same layer type used in reduction, but without stride (in this way,
        the operators which can already reshape both spatial and depth dimensions, i.e., non-dilated convolution, can just implement reduction).

        Args:
            match: regex Match object
            filters: desired number of output channels
            name_prefix: an optional prefix for layer name
            name_suffix: an optional suffix for layer name

        Returns:
            Keras layer implementing the POPNAS operator, without shape modifications between input and output
        '''
        return self.generate_reduction_layer(match, filters, tuple([1] * self.op_dims), name_prefix, name_suffix)

    def generate_depth_adaptation_layer(self, match: re.Match, filters: int, name_prefix: str = 'D/', name_suffix: str = '') -> Layer:
        '''
        Generate a layer for the operator, for cases where only the depth (channels) changes between input and output.
        If not overridden, then the function generates the same layer type used in reduction, but without stride (in this way,
        the operators which can already reshape both spatial and depth dimensions, i.e., non-dilated convolution, can just implement reduction).

        Args:
            match: regex Match object
            filters: desired number of output channels
            name_prefix: an optional prefix for layer name
            name_suffix: an optional suffix for layer name

        Returns:
            Keras layer implementing the POPNAS operator, with shape modifications between input and output
        '''
        return self.generate_reduction_layer(match, filters, tuple([1] * self.op_dims), name_prefix, name_suffix)

    # TODO: not relevant right now, since there aren't cases where only spatial adaptation happens, but makes the API flexible for future changes
    def generate_spatial_adaptation_layer(self, match: re.Match, filters: int, strides: 'tuple[int, ...]',
                                          name_prefix: str = 'S/', name_suffix: str = '') -> Layer:
        '''
        Generate a layer for the operator, for cases where only the spatial dimensions change between input and output.
        If not overridden, then the function generates the same layer type used in reduction (in this way,
        the operators which can already reshape both spatial and depth dimensions, i.e., non-dilated convolution, can just implement reduction).

        Args:
            match: regex Match object
            filters: desired number of output channels
            strides: the strides to apply
            name_prefix: an optional prefix for layer name
            name_suffix: an optional suffix for layer name

        Returns:
            Keras layer implementing the POPNAS operator, with shape modifications between input and output
        '''
        raise self.generate_reduction_layer(match, filters, strides, name_prefix, name_suffix)

    @abstractmethod
    def generate_reduction_layer(self, match: re.Match, filters: int, strides: 'tuple[int, ...]',
                                 name_prefix: str = 'R/', name_suffix: str = '') -> Layer:
        '''
        Generate a reduction layer for the operator (both spatial or depth dimensionality changes).

        Args:
            match: regex Match object
            filters: desired number of output channels
            strides: the strides to apply
            name_prefix: an optional prefix for layer name
            name_suffix: an optional suffix for layer name

        Returns:
            Keras layer implementing the POPNAS operator, with shape modifications between input and output
        '''
        raise NotImplementedError()
