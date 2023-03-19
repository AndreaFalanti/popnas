import math
from typing import Optional, Callable

from tensorflow.keras import activations
from tensorflow.keras.layers import Layer
from tensorflow.keras.regularizers import Regularizer

from ..layers import op_dim_selector


class SqueezeExcitation(Layer):
    ''' Layer implementing the *Squeeze-and-Excitation* block, presented in the "Squeeze-and-Excitation Networks" paper. '''

    def __init__(self, dims: int, filters: int, se_ratio: int, use_bias: bool, weight_reg: Optional[Regularizer] = None,
                 activation_f: Callable = activations.swish, name='squeeze_excitation', **kwargs):
        ''' Creates a squeeze and excitation layer. *Dims* parameter is used to adapt to either images (2D) or time series (1D) tensor shapes. '''
        super().__init__(name=name, **kwargs)

        self.dims = dims
        self.filters = filters
        self.se_ratio = se_ratio
        self.use_bias = use_bias
        self.weight_reg = weight_reg
        self.activation_f = activation_f

        bottleneck_filters = math.ceil(filters / se_ratio)
        conv = op_dim_selector['conv'][dims]
        gap = op_dim_selector['gap'][dims]

        self.gap = gap(keepdims=True)
        # use pointwise convolutions for densely connected; they are equivalent since the volume is in shape (1, 1, C)
        # use directly the Keras conv layers, since we don't want batch normalization here
        self.first_dc = conv(bottleneck_filters, kernel_size=1, activation=activation_f, use_bias=use_bias,
                             kernel_initializer='VarianceScaling', kernel_regularizer=weight_reg)
        # the second "dense" activation is fixed to sigmoid by layer definition
        self.second_dc = conv(filters, kernel_size=1, activation=activations.sigmoid, use_bias=use_bias,
                              kernel_initializer='VarianceScaling', kernel_regularizer=weight_reg)

    def call(self, inputs, **kwargs):
        x = self.gap(inputs)
        x = self.first_dc(x)
        x = self.second_dc(x)
        return x * inputs

    def get_config(self):
        config = super().get_config()
        config.update({
            'dims': self.dims,
            'filters': self.filters,
            'se_ratio': self.se_ratio,
            'use_bias': self.use_bias,
            'weight_reg': self.weight_reg,
            'activation_f': self.activation_f
        })
        return config


class ResizableSqueezeExcitation(Layer):
    ''' Layer which allows to resize the tensor shape inside a *Squeeze-and-Excitation* block,
     thanks to a preliminary layer (e.g. pooling) executed before the SE block. '''

    def __init__(self, se_layer: SqueezeExcitation, resize_layer: Layer, name='resized_squeeze_excitation', **kwargs):
        super().__init__(name=name, **kwargs)

        self.se_layer = se_layer
        self.resize_layer = resize_layer

    def call(self, inputs, **kwargs):
        x = self.resize_layer(inputs)
        return self.se_layer(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            'se_layer': self.se_layer,
            'resize_layer': self.resize_layer
        })
        return config
