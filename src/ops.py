import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, SeparableConv2D, MaxPooling2D, AveragePooling2D, BatchNormalization, Layer

# TODO: could use a pointwise convolution, like pooling. Experiment which one is better.


def depth_zero_pad_closure(desired_depth, op_layer):
    '''
    Pad depth of a Keras layer with zeros, if it has not already the right depth for performing addition at the end of the block.

    Args:
        desired_depth (int): Depth required for addition with other tensor inside the block (filters value).
        op_layer (tf.keras.Layer): Keras layer to use. Only layers that don't modify depth can be used, otherwise the closure will fail
            (as it wouldn't be necessary to adapt the depth).

    Returns:
        (Callable): call function build by closure
    '''
    def depth_zero_pad_call(inputs):
        # directly compute depth on input, because 
        input_depth = inputs.get_shape().as_list()[3]
        pad_size = desired_depth - input_depth

        if pad_size > 0:
            output = op_layer(inputs)
            paddings = tf.constant([[0, 0], [0, 0], [0, 0], [0, pad_size]])
            return tf.pad(output, paddings)  # pad output with 0s
        else:
            return op_layer(inputs)

    return depth_zero_pad_call


def depth_pointwise_conv_closure(desired_depth, op_layer):
    '''
    Generate call for pooling operation. It adds a pointwise convolution to adapt the depth, if necessary.

    Args:
        desired_depth (int): Depth required for addition with other tensor inside the block (filters value).
        op_layer (tf.keras.Layer): Keras layer to use. Only layers that don't modify depth can be used, otherwise the closure will fail
            (as it wouldn't be necessary to adapt the depth).

    Returns:
        (Callable): call function build by closure
    '''
    def depth_pointwise_conv_call(inputs):
        input_depth = inputs.get_shape().as_list()[3]

        if input_depth != desired_depth:
            output = op_layer(inputs)
            return Convolution(desired_depth, (1, 1), (1, 1))(output)
        else:
            return op_layer(inputs)

    return depth_pointwise_conv_call


class Identity(Model):

    def __init__(self, filters, strides):
        '''
        Simply adds an identity connection, padding in depth with 0 if necessary to enable block add operator.
        '''
        super(Identity, self).__init__()

        self.op = Layer()   # Identity layer in Keras
        self.identity_call = depth_zero_pad_closure(filters, self.op)

    def call(self, inputs, training=None, mask=None):
        return self.identity_call(inputs)


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

        self.pool_call = depth_pointwise_conv_closure(filters, self.pool)

    def call(self, inputs, training=None, mask=None):
        return self.pool_call(inputs)
