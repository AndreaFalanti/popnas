import operator
from functools import reduce
from typing import Iterable, Sequence

import tensorflow as tf


# TODO: not used anymore for simplicity, but is still a great utility function if will be needed in the future
def compute_tensor_byte_size(tensor: tf.Tensor):
    dtype_sizes = {
        tf.float16: 2,
        tf.float32: 4,
        tf.float64: 8,
        tf.int32: 4,
        tf.int64: 8
    }

    dtype_size = dtype_sizes[tensor.dtype]
    # remove batch size from shape
    tensor_shape = tensor.get_shape().as_list()[1:]

    # byte size is: (number of weights) * (size of each weight)
    return reduce(operator.mul, tensor_shape, 1) * dtype_size


def compute_bytes_from_tensor_shape(tensor_shape: Iterable[int], dtype_byte_size: int):
    # byte size is: (number of weights) * (size of each weight)
    return reduce(operator.mul, tensor_shape, 1) * dtype_byte_size


def have_tensors_same_spatial(first_shape: Sequence[float], second_shape: Sequence[float]):
    return first_shape[:-1] == second_shape[:-1]


def have_tensors_same_depth(first_shape: Sequence[float], second_shape: Sequence[float]):
    return first_shape[-1] == second_shape[-1]


def alter_tensor_shape(shape: Sequence[float], spatial_mult: float = 1, depth_mult: float = 1):
    return [el * spatial_mult for el in shape[:-1]] + [shape[-1] * depth_mult]


def get_tensors_spatial_ratio(first_shape: Sequence[float], second_shape: Sequence[float]):
    return first_shape[0] / second_shape[0]
