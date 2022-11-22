import math
from typing import Optional

from tensorflow.keras import activations
from tensorflow.keras.layers import Layer
from tensorflow.keras.regularizers import Regularizer

from ..layers import op_dim_selector


class SqueezeExcitation(Layer):
    ''' Layer implementing the *Squeeze-and-Excitation* block, presented in the "Squeeze-and-Excitation Networks" paper. '''

    def __init__(self, dims: int, filters: int, se_ratio: int, use_bias: bool, weight_reg: Optional[Regularizer] = None,
                 name='squeeze_excitation', **kwargs):
        ''' Creates a squeeze and excitation layer. *Dims* parameter is used to adapt to either images (2D) or time series (1D) tensor shapes. '''
        super().__init__(name=name, **kwargs)

        self.dims = dims
        self.filters = filters
        self.se_ratio = se_ratio
        self.use_bias = use_bias
        self.weight_reg = weight_reg

        bottleneck_filters = math.ceil(filters / se_ratio)
        conv = op_dim_selector['conv'][dims]
        gap = op_dim_selector['gap'][dims]

        self.gap = gap(keepdims=True)
        # use pointwise convolutions for densely connected, they are equivalent since the volume is in shape (1, 1, C)
        self.first_dc = conv(bottleneck_filters, kernel_size=1, activation=activations.swish, use_bias=use_bias,
                             kernel_initializer='VarianceScaling', kernel_regularizer=weight_reg)
        self.second_dc = conv(filters, kernel_size=1, activation=activations.sigmoid, use_bias=use_bias,
                              kernel_initializer='VarianceScaling', kernel_regularizer=weight_reg)

    def call(self, inputs):
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
        })
        return config
