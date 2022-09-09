import operator
from abc import abstractmethod, ABC

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, SeparableConv2D, MaxPooling2D, AveragePooling2D, \
    Conv1D, Conv1DTranspose, SeparableConv1D, MaxPooling1D, AveragePooling1D, \
    BatchNormalization, Layer

# TODO: it could be nice to import only the correct ones and rename them so that are recognized by layers,
#  but input dims are given at runtime, making it difficult (inner classes with delayed import seems not possible)
op_dim_selector = {
    'conv': {1: Conv1D, 2: Conv2D},
    'tconv': {1: Conv1DTranspose, 2: Conv2DTranspose},
    'dconv': {1: SeparableConv1D, 2: SeparableConv2D},
    'max_pool': {1: MaxPooling1D, 2: MaxPooling2D},
    'avg_pool': {1: AveragePooling1D, 2: AveragePooling2D},
}


class Identity(Layer):
    def __init__(self, name='identity', **kwargs):
        '''
        Simply adds an identity connection.
        '''
        super().__init__(name=name, **kwargs)

    def call(self, inputs, training=None, mask=None):
        return tf.identity(inputs)


# TODO: deprecated in previous versions, even if it should work fine for 2D. Right now it needs a refactor to adapt to different input dims.
# class IdentityReshaper(Layer):
#     def __init__(self, filters, input_filters, strides, name='identity_reshaper', **kwargs):
#         '''
#         Identity alternative when the tensor shape between input and output differs.
#         IdentityReshaper can apply a stride without doing any operation,
#         also adapting depth by replicating multiple times the tensor and concatenating the replicas on depth axis.
#         '''
#         super().__init__(name=name, **kwargs)
#         self.filters = filters
#         self.input_filters = input_filters
#         self.strides = strides
#         self.no_stride = self.strides[0] == 1 and self.strides[1] == 1
#
#         self.replication_factor = math.ceil(filters / input_filters)
#
#     def call(self, inputs, training=None, mask=None):
#         # TODO: it seems that ::1 slice is bugged, if stride is used when not needed the model has random output. TF bug?
#         input_stride = inputs if self.no_stride else inputs[:, ::self.strides[0], ::self.strides[1], :]
#         return tf.tile(input_stride, [1, 1, 1, self.replication_factor])[:, :, :, :self.filters]
#
#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             'filters': self.filters,
#             'input_filters': self.input_filters,
#             'strides': self.strides
#         })
#         return config


class OpBatchActivation(Layer, ABC):
    @abstractmethod
    def __init__(self, filters, kernel, strides, weight_reg=None, name='abstract', **kwargs):
        '''
        Abstract utility class used as baseline for any {Operation - Batch Normalization - Activation} layer.
        Op attribute must be set to a Keras layer or TF nn operation in all concrete implementations.
        '''
        super().__init__(name=name, **kwargs)
        self.filters = filters
        self.kernel = kernel
        self.strides = strides
        self.weight_reg = weight_reg

        self.bn = BatchNormalization()

        # concrete implementations must use a valid Keras layer / TF operation, assigning it to this variable during __init__
        self.op = None

    def call(self, inputs, training=None, mask=None):
        x = self.op(inputs)
        x = self.bn(x, training=training)
        return tf.nn.silu(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel': self.kernel,
            'strides': self.strides,
            'weight_reg': self.weight_reg
        })
        return config


class SeparableConvolution(OpBatchActivation):
    def __init__(self, filters, kernel, strides, weight_reg=None, name='dconv', **kwargs):
        '''
        Constructs a {Separable Convolution - Batch Normalization - Activation} layer.
        '''
        super().__init__(filters, kernel, strides, weight_reg, name=name, **kwargs)

        dconv = op_dim_selector['dconv'][len(kernel)]
        self.op = dconv(filters, kernel, strides=strides, padding='same',
                        depthwise_initializer='he_uniform', pointwise_initializer='he_uniform',
                        depthwise_regularizer=weight_reg, pointwise_regularizer=weight_reg)


class Convolution(OpBatchActivation):
    def __init__(self, filters, kernel, strides, weight_reg=None, name='conv', **kwargs):
        '''
        Constructs a {Spatial Convolution - Batch Normalization - Activation} layer.
        '''
        super().__init__(filters, kernel, strides, weight_reg, name=name, **kwargs)

        conv = op_dim_selector['conv'][len(kernel)]
        self.op = conv(filters, kernel, strides=strides, padding='same',
                       kernel_initializer='he_uniform', kernel_regularizer=weight_reg)


class TransposeConvolution(OpBatchActivation):
    def __init__(self, filters, kernel, strides, weight_reg=None, name='tconv', **kwargs):
        '''
        Constructs a {Transpose Convolution - Batch Normalization - Activation} layer.
        '''
        super().__init__(filters, kernel, strides, weight_reg, name=name, **kwargs)

        tconv = op_dim_selector['tconv'][len(kernel)]
        self.op = tconv(filters, kernel, strides, padding='same',
                        kernel_initializer='he_uniform', kernel_regularizer=weight_reg)


class TransposeConvolutionStack(Layer):
    def __init__(self, filters, kernel, strides, weight_reg=None, name='tconv', **kwargs):
        '''
        Constructs a Transpose Convolution - Convolution layer. Batch Normalization and Relu are applied on both.
        '''
        super().__init__(name=name, **kwargs)
        self.filters = filters
        self.kernel = kernel
        self.strides = strides
        self.weight_reg = weight_reg

        transpose_stride = tuple([2] * len(strides))
        conv_strides = tuple(map(operator.mul, transpose_stride, strides))

        self.transposeConv = TransposeConvolution(filters, kernel, transpose_stride, weight_reg)
        self.conv = Convolution(filters, kernel, conv_strides, weight_reg)

    def call(self, inputs, training=None, mask=None):
        x = self.transposeConv(inputs)
        return self.conv(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel': self.kernel,
            'strides': self.strides,
            'weight_reg': self.weight_reg
        })
        return config


class StackedConvolution(Layer):
    def __init__(self, filter_list, kernel_list, stride_list, weight_reg=None, name='stack_conv', **kwargs):
        '''
        Constructs a stack of Convolution blocks that are chained together.
        '''
        super().__init__(name=name, **kwargs)
        self.filter_list = filter_list
        self.kernel_list = kernel_list
        self.stride_list = stride_list
        self.weight_reg = weight_reg

        assert len(filter_list) == len(kernel_list) and len(kernel_list) == len(stride_list), "List lengths must match"

        self.convs = [Convolution(f, k, s, weight_reg=weight_reg) for (f, k, s) in zip(filter_list, kernel_list, stride_list)]

    def call(self, inputs, training=None, mask=None):
        x = inputs

        for conv in self.convs:
            x = conv(x, training=training)

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'filter_list': self.filter_list,
            'kernel_list': self.kernel_list,
            'stride_list': self.stride_list,
            'weight_reg': self.weight_reg
        })
        return config


class Pooling(Layer):
    def __init__(self, pool_type, size, strides, name='pool', **kwargs):
        '''
        Constructs a standard pooling layer (average or max).
        '''
        super().__init__(name=name, **kwargs)
        self.pool_type = pool_type
        self.size = size
        self.strides = strides

        pool_layer = op_dim_selector[f'{pool_type}_pool'][len(size)]
        self.pool = pool_layer(size, strides, padding='same')

    def call(self, inputs, training=None, mask=None):
        return self.pool(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({
            'pool_type': self.pool_type,
            'size': self.size,
            'strides': self.strides
        })
        return config


class PoolingConv(Layer):
    def __init__(self, filters, pool_type, size, strides, weight_reg=None, name='pool_conv', **kwargs):
        '''
        Constructs a pooling layer (average or max). It also adds a pointwise convolution (using Convolution class)
        to adapt the output depth to filters size.
        '''
        super().__init__(name=name, **kwargs)
        self.filters = filters
        self.pool_type = pool_type
        self.size = size
        self.strides = strides
        self.weight_reg = weight_reg

        self.pool = Pooling(pool_type, size, strides)

        ones = tuple([1] * len(size))
        self.pointwise_conv = Convolution(filters, kernel=ones, strides=ones, weight_reg=weight_reg)

    def call(self, inputs, training=None, mask=None):
        output = self.pool(inputs)
        return self.pointwise_conv(output)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'pool_type': self.pool_type,
            'size': self.size,
            'strides': self.strides,
            'weight_reg': self.weight_reg
        })
        return config


class ScalarMult(Layer):
    def __init__(self, scalar: float, name='scalar_mult', **kwargs):
        '''
        Multiply tensor for a scalar. Wrapper of tf.math.scalar_mul.
        '''
        super().__init__(name=name, **kwargs)
        self.scalar = scalar

    def call(self, inputs, training=None, mask=None):
        return tf.math.scalar_mul(self.scalar, inputs)

    def get_config(self):
        config = super().get_config()
        config.update({
            'scalar': self.scalar,
        })
        return config
