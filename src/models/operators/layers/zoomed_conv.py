from typing import Optional, Callable

from tensorflow.keras import activations
from tensorflow.keras.layers import Layer
from tensorflow.keras.regularizers import Regularizer

from models.operators.layers.common import Convolution, Pooling
from ..layers import op_dim_selector


class ZoomedConvolution(Layer):
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

        op_dims = len(kernel)
        pool_size_and_stride = tuple([zoom_factor] * op_dims)

        self.downsampler = Pooling('avg', pool_size_and_stride, pool_size_and_stride)
        self.conv = Convolution(filters, kernel, strides, weight_reg=weight_reg, activation_f=activation_f)

        upsampler = op_dim_selector['upsample'][op_dims]
        # TODO: currently Upsample1D is the one implemented in Keras, which does not support linear upsample.
        #  Refer to this issue for extending it: https://github.com/tensorflow/tensorflow/issues/46609
        # TODO: has problems in case an input dimension is odd, because it produce 1 value more than input on these axes...
        self.upsampler = upsampler(pool_size_and_stride, interpolation='bilinear', name=name) if op_dims == 2 \
            else upsampler(zoom_factor, name=name)

    def call(self, inputs, training=None, mask=None):
        x = self.downsampler(inputs)
        x = self.conv(x, training=training)
        return self.upsampler(x)

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
