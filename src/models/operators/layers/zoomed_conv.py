from abc import ABC
from typing import Optional, Callable

import tensorflow as tf
from tensorflow.keras import activations
from tensorflow.keras.layers import Layer
from tensorflow.keras.regularizers import Regularizer

from models.operators.layers.common import Convolution, Pooling
from ..layers import op_dim_selector


class BaseZoomedConvolution(Layer, ABC):
    def __init__(self, filters: int, kernel: 'tuple[int, ...]', strides: 'tuple[int, ...]', zoom_factor: int,
                 weight_reg: Optional[Regularizer] = None, activation_f: Callable = activations.swish, name='zconv', **kwargs):
        '''
        Zoomed convolution presented in FasterSeg paper.

        An alternative way of expanding the receptive field of convolutions without increasing the kernel size,
        by performing (downsampling, conv, upsampling) in this order.

        Should be more efficient in terms on FLOPs and inference time compared to dilated convolutions with the same factor.
        '''
        super().__init__(name=name, **kwargs)
        self.filters = filters
        self.kernel = kernel
        self.strides = strides
        self.zoom_factor = zoom_factor
        self.weight_reg = weight_reg
        self.activation_f = activation_f

        self.op_dims = len(kernel)
        self.pool_size = tuple([zoom_factor] * self.op_dims)

        self.downsampler = Pooling('avg', self.pool_size, self.pool_size)
        self.conv = Convolution(filters, kernel, strides, weight_reg=weight_reg, activation_f=activation_f)

        self.upsampler_class = op_dim_selector['upsample'][self.op_dims]

        # add batch size and channels to dims count
        # self.slice_start = tf.zeros(self.op_dims + 2, tf.int32)

    def build(self, input_shape: tf.TensorShape):
        # discard batch size and channels, retrieving only the resolution
        input_res = tf.constant(input_shape[1:-1])
        # divide by stride factor, ceiling the value (solves odd-sized axes shape discrepancy), and casting back to int
        self.output_resolution = tf.cast(tf.math.ceil(input_res / self.strides[0]), tf.int32)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel': self.kernel,
            'strides': self.strides,
            'zoom_factor': self.dilation_rate,
            'weight_reg': self.weight_reg,
            'activation_f': self.activation_f
        })
        return config


class ZoomedConvolution2D(BaseZoomedConvolution):
    def __init__(self, filters: int, kernel: 'tuple[int, ...]', strides: 'tuple[int, ...]', zoom_factor: int,
                 weight_reg: Optional[Regularizer] = None, activation_f: Callable = activations.swish, name='zconv', **kwargs):
        super().__init__(filters, kernel, strides, zoom_factor, weight_reg, activation_f, name=name, **kwargs)

        self.upsampler = self.upsampler_class(self.pool_size, interpolation='bilinear', name=name)

    def call(self, inputs, training=None, mask=None):
        x = self.downsampler(inputs)
        x = self.conv(x, training=training)
        x = self.upsampler(x)
        # force the output to have the same resolution of the input, or half of it in case of reduction
        # solves the shape problem with odd dimensions (upsample has +1 element on odd-sized axes compared to the input)
        return x[:, :self.output_resolution[0], :self.output_resolution[1], :]


class ZoomedConvolution1D(BaseZoomedConvolution):
    def __init__(self, filters: int, kernel: 'tuple[int, ...]', strides: 'tuple[int, ...]', zoom_factor: int,
                 weight_reg: Optional[Regularizer] = None, activation_f: Callable = activations.swish, name='zconv', **kwargs):
        super().__init__(filters, kernel, strides, zoom_factor, weight_reg, activation_f, name=name, **kwargs)

        # TODO: currently Upsample1D is the one implemented in Keras, which does not support linear upsample.
        #  Refer to this issue for extending it: https://github.com/tensorflow/tensorflow/issues/46609
        self.upsampler = self.upsampler_class(zoom_factor, name=name)

    def call(self, inputs, training=None, mask=None):
        x = self.downsampler(inputs)
        x = self.conv(x, training=training)
        x = self.upsampler(x)
        # force the output to have the same resolution of the input, or half of it in case of reduction
        # solves the shape problem with odd dimensions (upsample has +1 element on odd-sized axes compared to the input)
        return x[:, :self.output_resolution[0], :]


# The main class to use in the OP allocator.
# Basically, it wraps the ZoomedConvolution layer working on the dimensionality of the input.
class ZoomedConvolution(Layer):
    def __init__(self, filters: int, kernel: 'tuple[int, ...]', strides: 'tuple[int, ...]', zoom_factor: int,
                 weight_reg: Optional[Regularizer] = None, activation_f: Callable = activations.swish, name='zconv', **kwargs):
        super().__init__(name=name, **kwargs)
        op_dims = len(kernel)

        if op_dims == 1:
            self.zconv = ZoomedConvolution1D(filters, kernel, strides, zoom_factor, weight_reg, activation_f, name, **kwargs)
        elif op_dims == 2:
            self.zconv = ZoomedConvolution2D(filters, kernel, strides, zoom_factor, weight_reg, activation_f, name, **kwargs)
        else:
            raise ValueError('Zoomed convolution not implemented for 3D domain')

    def call(self, inputs, training=None, mask=None):
        return self.zconv(inputs, training, mask)

    def get_config(self):
        return self.zconv.get_config()


