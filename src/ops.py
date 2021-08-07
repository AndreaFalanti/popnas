import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, SeparableConv2D, MaxPooling2D, AveragePooling2D, BatchNormalization

# TODO: could use a pointwise convolution, like pooling. Experiment which one is better.


def pad_closure(desired_depth):
    '''
    Pad depth of an identity tensor with zeros if has not already the right depth for performing addition at the end of the block.

    Args:
        desired_depth (int): Depth required for addition with other tensor inside the block

    Returns:
        (Callable): call function build by closure
    '''
    def pad_identity(inputs):
        input_depth = inputs.get_shape().as_list()[3]
        pad_size = desired_depth - input_depth

        if pad_size > 0:
            paddings = tf.constant([[0, 0], [0, 0], [0, 0], [0, pad_size]])
            return tf.pad(inputs, paddings)  # constant, with 0s
        else:
            return inputs

    return pad_identity


def pooling_closure(filters, pool_layer):
    '''
    Generate call for pooling operation. It adds a pointwise convolution to adapt the depth, if necessary.

    Args:
        filters (int): number of filters (depth)
        pool_layer (tf.keras.Layer): pooling layer to use (max or avg already parametrized)

    Returns:
        (Callable): call function build by closure
    '''
    def pooling_call(inputs):
        input_depth = inputs.get_shape().as_list()[3]

        if input_depth != filters:
            pool = pool_layer(inputs)
            return Convolution(filters, (1, 1), (1, 1))(pool)
        else:
            return pool_layer(inputs)

    return pooling_call


class Identity(Model):

    def __init__(self, filters, strides):
        '''
        Simply adds an identity connection, padding in depth with 0 if necessary to enable block add operator.
        '''
        super(Identity, self).__init__()

        self.op = pad_closure(filters)

    def call(self, inputs, training=None, mask=None):
        return self.op(inputs)


class SeperableConvolution(Model):

    def __init__(self, filters, kernel, strides):
        '''
        Constructs a Seperable Convolution - Batch Normalization - Relu block.
        '''
        super(SeperableConvolution, self).__init__()

        self.conv = SeparableConv2D(filters, kernel, strides=strides, padding='same',
                                    kernel_initializer='he_uniform')
        self.bn = BatchNormalization()

    def call(self, inputs, training=None, mask=None):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        return tf.nn.relu(x)


class Convolution(Model):

    def __init__(self, filters, kernel, strides):
        '''
        Constructs a Spatial Convolution - Batch Normalization - Relu block.
        '''
        super(Convolution, self).__init__()

        self.conv = Conv2D(filters, kernel, strides=strides, padding='same',
                           kernel_initializer='he_uniform')
        self.bn = BatchNormalization()

    def call(self, inputs, training=None, mask=None):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        return tf.nn.relu(x)


class StackedConvolution(Model):

    def __init__(self, filter_list, kernel_list, stride_list):
        '''
        Constructs a stack of Convolution blocks that are chained together.
        '''
        super(StackedConvolution, self).__init__()

        assert len(filter_list) == len(kernel_list) and len(kernel_list) == len(stride_list), "List lengths must match"

        self.convs = []
        for i, (f, k, s) in enumerate(zip(filter_list, kernel_list, stride_list)):
            conv = Convolution(f, k, s)
            self.convs.append(conv)

    def call(self, inputs, training=None, mask=None):
        x = inputs

        for conv in self.convs:
            x = conv(x, training=training)

        return x


class Pooling(Model):

    def __init__(self, filters, type, size, strides):
        '''
        Constructs a pooling layer (average or max). It also adds a pointwise convolution (using Convolution class) to adapt
        the output depth to filters size, if they are not the same.
        '''
        super(Pooling, self).__init__()
        if type == 'max':
            self.pool = MaxPooling2D(size, strides, padding='same')
        else:
            self.pool = AveragePooling2D(size, strides, padding='same')

        self.pool_call = pooling_closure(filters, self.pool)

    def call(self, inputs, training=None, mask=None):
        return self.pool_call(inputs)
