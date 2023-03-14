from typing import Optional, Callable

import tensorflow as tf
from tensorflow.keras.layers import Layer, GlobalAveragePooling2D
from tensorflow.keras.regularizers import Regularizer

from models.operators.layers import Convolution


class ImagePooling(Layer):
    def __init__(self, filters: int, weight_reg: Optional[Regularizer] = None, activation_f: Callable = tf.nn.silu, name='img_pooling', **kwargs):
        '''
        Image global pooling used in ASPP module.
        The implementation is different from the one proposed in the ParseNet paper, using batch normalization instead of L2 norm,
         and pointwise convolution to adapt the filters.
        '''
        super().__init__(name=name, **kwargs)
        self.filters = filters
        self.weight_reg = weight_reg
        self.activation_f = activation_f

        self.gap = GlobalAveragePooling2D(keepdims=True)
        self.pw_conv = Convolution(filters, kernel=(1, 1), strides=(1, 1), weight_reg=weight_reg, activation_f=activation_f)

    def call(self, inputs, training=None, mask=None):
        x = self.gap(inputs)
        x = self.pw_conv(x)
        return tf.image.resize(x, tf.shape(inputs)[1:3])   # bilinear upsample to original dimensionality

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'weight_reg': self.weight_reg,
            'activation_f': self.activation_f
        })
        return config


class AtrousSpatialPyramidPooling(Layer):
    def __init__(self, filters: int, dilation_rates: 'tuple[int, int, int]', filters_ratio: float = 1,
                 weight_reg: Optional[Regularizer] = None, activation_f: Callable = tf.nn.silu, name='ASPP', **kwargs):
        '''
        ASPP layer.
        The implementation is inspired from: https://github.com/tensorflow/models/blob/v2.11.3/official/vision/modeling/layers/deeplab.py
        '''
        super().__init__(name=name, **kwargs)
        self.filters = filters
        self.dilation_rates = dilation_rates
        self.weight_reg = weight_reg
        self.activation_f = activation_f
        self.filters_ratio = filters_ratio

        op_filters = int(filters * filters_ratio)

        self.conv_1x1 = Convolution(op_filters, kernel=(1, 1), strides=(1, 1), weight_reg=weight_reg, activation_f=activation_f)
        self.conv_3x3_1 = Convolution(op_filters, kernel=(3, 3), strides=(1, 1), dilation_rate=dilation_rates[0],
                                      weight_reg=weight_reg, activation_f=activation_f)
        self.conv_3x3_2 = Convolution(op_filters, kernel=(3, 3), strides=(1, 1), dilation_rate=dilation_rates[1],
                                      weight_reg=weight_reg, activation_f=activation_f)
        self.conv_3x3_3 = Convolution(op_filters, kernel=(3, 3), strides=(1, 1), dilation_rate=dilation_rates[2],
                                      weight_reg=weight_reg, activation_f=activation_f)
        self.image_pool = ImagePooling(op_filters, weight_reg=weight_reg, activation_f=activation_f)

        self.bottleneck = Convolution(filters, kernel=(1, 1), strides=(1, 1), weight_reg=weight_reg, activation_f=activation_f)

    def call(self, inputs, training=None, mask=None):
        x1 = self.conv_1x1(inputs)
        x2 = self.conv_3x3_1(inputs)
        x3 = self.conv_3x3_2(inputs)
        x4 = self.conv_3x3_3(inputs)
        x5 = self.image_pool(inputs)

        x = tf.concat([x1, x2, x3, x4, x5], axis=-1)
        return self.bottleneck(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'dilation_rates': self.dilation_rates,
            'filters_ratio': self.filters_ratio,
            'weight_reg': self.weight_reg,
            'activation_f': self.activation_f
        })
        return config
